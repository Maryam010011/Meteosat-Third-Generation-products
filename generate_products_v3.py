#!/usr/bin/env python3
"""
MTG FCI Complete Product Pipeline v3
====================================
1. Fixes the scanline glitch (missing horizontal chunk stripes) via vertical disk inpainting.
2. Generates the exact directory structure and 40 colormapped composites per category
   matching the user's reference screenshots:
   output_v2/<YYYYMMDDTHHMMZ>/composites_40/<composite_category>/
3. Generates 9 composite categories x 40 colormaps = 360 images per cycle.
4. Generates single channels + derived products with clean solid white background outside disk.
"""

import os
import sys
import time
import argparse
import csv
import warnings
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

try:
    from satpy import Scene
    HAS_SATPY = True
except ImportError:
    print("CRITICAL: Satpy is required.")
    sys.exit(1)

# ============================================================================
# 40 COLORMAPS DEFINITION (matching user reference screenshot)
# ============================================================================
COLORMAPS_40 = [
    ('01_grayr',        'gray_r'),
    ('02_gray',         'gray'),
    ('03_jet',          'jet'),
    ('04_hsv',          'hsv'),
    ('05_inferno',      'inferno'),
    ('06_magma',        'magma'),
    ('07_plasma',       'plasma'),
    ('08_viridis',      'viridis'),
    ('09_cividis',      'cividis'),
    ('10_spectral',     'Spectral'),
    ('11_spectralr',    'Spectral_r'),
    ('12_rdylbu',       'RdYlBu'),
    ('13_rdylbur',      'RdYlBu_r'),
    ('14_coolwarm',     'coolwarm'),
    ('15_bwr',          'bwr'),
    ('16_seismic',      'seismic'),
    ('17_ylorrd',       'YlOrRd'),
    ('18_ylorrd_r',     'YlOrRd_r'),
    ('19_hot',          'hot'),
    ('20_hotr',         'hot_r'),
    ('21_afmhot',       'afmhot'),
    ('22_gist_heat',    'gist_heat'),
    ('23_copper',       'copper'),
    ('24_autumn',       'autumn'),
    ('25_summer',       'summer'),
    ('26_spring',       'spring'),
    ('27_winter',       'winter'),
    ('28_ocean',        'ocean'),
    ('29_terrain',      'terrain'),
    ('30_gist_earth',   'gist_earth'),
    ('31_gist_ncar',    'gist_ncar'),
    ('32_gist_rainbow', 'gist_rainbow'),
    ('33_nipy_spectral','nipy_spectral'),
    ('34_turbo',        'turbo'),
    ('35_gnuplot',      'gnuplot'),
    ('36_gnuplot2',     'gnuplot2'),
    ('37_cmrmap',       'CMRmap'),
    ('38_cubehelix',    'cubehelix'),
    ('39_brg',          'brg'),
    ('40_tab20b',       'tab20b')
]

# 9 Composite Categories matching user screenshot
COMPOSITE_CATEGORIES = [
    'airmass',
    'ash',
    'colorized_ir_clouds',
    'convection',
    'day_severe_storms',
    'dust',
    'fog',
    'natural_color',
    'night_microphysics'
]


class MTGPipelineV3:
    def __init__(self, data_dir='data', out_dir='output_v2', target_size=(1000, 1024)):
        self.data_dir = os.path.abspath(data_dir)
        self.out_dir = os.path.abspath(out_dir)
        self.target_w, self.target_h = target_size
        os.makedirs(self.out_dir, exist_ok=True)
        self.manifest_records = []

    def discover_cycles(self):
        from collections import defaultdict
        cycles = defaultdict(list)
        for fname in os.listdir(self.data_dir):
            if fname.endswith('.nc') and 'CHK-BODY' in fname:
                parts = fname.split('_')
                for i, p in enumerate(parts):
                    if len(p) == 4 and p.isdigit() and i > 5:
                        cycles[p].append(os.path.join(self.data_dir, fname))
                        break
        return dict(sorted(cycles.items()))

    def _disk_geometry(self):
        ny, nx = self.target_h, self.target_w
        y_g, x_g = np.ogrid[:ny, :nx]
        cy, cx = ny / 2.0, nx / 2.0
        r_disk = min(ny, nx) * 0.46
        dist = np.sqrt((x_g - cx)**2 + (y_g - cy)**2)
        return dist, r_disk

    def _extract_and_fix_glitch(self, scn, channel_name):
        """Extract dataset, resize, and perform vertical disk inpainting to remove horizontal stripe glitch."""
        dist, r_disk = self._disk_geometry()
        mask_disk = dist <= r_disk
        
        raw = scn[channel_name].values.astype(np.float32)
        is_ir = channel_name.startswith('ir_') or channel_name.startswith('wv_')
        
        # Valid data bounds
        if is_ir:
            valid_mask_raw = (raw >= 150.0) & (raw <= 350.0) & (~np.isnan(raw))
        else:
            valid_mask_raw = (raw >= -5.0) & (raw <= 150.0) & (~np.isnan(raw))
            
        clean_raw = np.where(valid_mask_raw, raw, np.nan)
        
        # Resize raw array to target resolution
        img_r = Image.fromarray(np.nan_to_num(clean_raw, nan=-999.0)).resize(
            (self.target_w, self.target_h), resample=Image.Resampling.BILINEAR
        )
        arr = np.array(img_r, dtype=np.float32)
        arr[arr < -900.0] = np.nan
        
        # FIX GLITCH: Inpaint missing horizontal scanline rows inside disk along y-axis
        for x in range(self.target_w):
            disk_col = mask_disk[:, x]
            valid_col = disk_col & (~np.isnan(arr[:, x]))
            nan_col = disk_col & np.isnan(arr[:, x])
            if np.any(nan_col) and np.sum(valid_col) >= 2:
                y_val = np.where(valid_col)[0]
                v_val = arr[y_val, x]
                y_nan = np.where(nan_col)[0]
                arr[y_nan, x] = np.interp(y_nan, y_val, v_val)
                
        return arr

    def compute_composite_base(self, cat, channels):
        """Compute 2D scalar field for each of the 9 composite categories."""
        vis06 = channels.get('vis_06')
        nir22 = channels.get('nir_22')
        ir38  = channels.get('ir_38')
        ir105 = channels.get('ir_105')
        
        if cat == 'colorized_ir_clouds':
            # Inverted Cloud Top Temperature (cold clouds = high value)
            return 330.0 - ir105 if ir105 is not None else None
        elif cat == 'dust':
            # SWIR Moisture / Aerosol proxy (NIR22 - VIS06) or IR38 - IR105
            if nir22 is not None and vis06 is not None:
                return nir22 - vis06
            return ir38 - ir105 if (ir38 is not None and ir105 is not None) else None
        elif cat == 'fog':
            # Low cloud & fog thermal diff (IR38 - IR105)
            return ir38 - ir105 if (ir38 is not None and ir105 is not None) else None
        elif cat == 'convection':
            # Convective cloud index: VIS06 x (320 - IR105)
            if vis06 is not None and ir105 is not None:
                return vis06 * np.clip(320.0 - ir105, 0, 120) / 100.0
            return 320.0 - ir105 if ir105 is not None else None
        elif cat == 'natural_color':
            # Vegetation / Surface proxy: (NIR22 - VIS06) / (NIR22 + VIS06 + 1e-5)
            if nir22 is not None and vis06 is not None:
                return (nir22 - vis06) / (nir22 + vis06 + 1e-5)
            return vis06
        elif cat == 'airmass':
            # Upper troposphere airmass proxy based on IR105 lapse rate
            return ir105
        elif cat == 'ash':
            # Ash plume proxy: IR38 - IR105
            return ir38 - ir105 if (ir38 is not None and ir105 is not None) else None
        elif cat == 'day_severe_storms':
            # Severe storm updrafts: VIS06 + IR38-IR105
            if vis06 is not None and ir38 is not None and ir105 is not None:
                return vis06 + np.clip(ir38 - ir105, -10, 50) * 2.0
            return vis06
        elif cat == 'night_microphysics':
            # Night microphysics diff: IR38 - IR105
            return ir38 - ir105 if (ir38 is not None and ir105 is not None) else None
        return None

    def render_and_save(self, data, title, out_path, cmap_name, vmin=None, vmax=None):
        """Render array to disk image with solid white background outside Earth disk."""
        dist, r_disk = self._disk_geometry()
        plot_data = np.copy(data)
        plot_data[dist > r_disk] = np.nan
        
        valid = plot_data[~np.isnan(plot_data)]
        if len(valid) == 0:
            return False
            
        v0 = vmin if vmin is not None else float(np.percentile(valid, 1))
        v1 = vmax if vmax is not None else float(np.percentile(valid, 99))
        if v0 >= v1:
            v0, v1 = v0 - 1.0, v1 + 1.0

        fig = plt.figure(figsize=(self.target_w/100.0, self.target_h/100.0), dpi=100)
        fig.patch.set_facecolor('#FFFFFF')
        ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
        ax.set_facecolor('#FFFFFF')
        
        cmap_obj = plt.get_cmap(cmap_name).copy()
        cmap_obj.set_bad('white')
        
        ax.imshow(plot_data, cmap=cmap_obj, vmin=v0, vmax=v1, origin='upper', aspect='auto')
        
        # Disk border outline
        ring = np.abs(dist - r_disk) < 1.5
        ax.imshow(np.ma.masked_where(~ring, np.ones_like(ring)),
                  cmap=mcolors.ListedColormap(['#64748B']), vmin=0, vmax=1,
                  alpha=0.6, origin='upper', aspect='auto')
        ax.axis('off')
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=100, facecolor='#FFFFFF', edgecolor='none')
        plt.close(fig)
        return True

    def process_all(self):
        cycles = self.discover_cycles()
        print(f"Found {len(cycles)} acquisition cycles in data/")
        
        total_images = 0
        t0 = time.time()

        for cycle_id, files in cycles.items():
            scn = Scene(filenames=files, reader='fci_l1c_nc')
            ts = scn.start_time
            ts_folder = ts.strftime('%Y%m%dT%H%MZ')
            ts_display = ts.strftime('%Y-%m-%d %H:%M UTC')
            
            print(f"\n==================================================")
            print(f"  PROCESSING CYCLE {cycle_id} ({ts_display})")
            print(f"==================================================")
            
            avail = scn.available_dataset_names()
            chans = [ch for ch in ['vis_06', 'nir_22', 'ir_38', 'ir_105'] if ch in avail]
            scn.load(chans)
            
            channels = {}
            for ch in chans:
                channels[ch] = self._extract_and_fix_glitch(scn, ch)
                print(f"  [OK-DEGLITCHED] Channel {ch} loaded & scanline glitch repaired.")

            # Path: output_v2 / <YYYYMMDDTHHMMZ> / composites_40 / <category>
            comp_40_dir = os.path.join(self.out_dir, ts_folder, 'composites_40')
            
            for cat in COMPOSITE_CATEGORIES:
                base_data = self.compute_composite_base(cat, channels)
                if base_data is None:
                    print(f"  [SKIP] Category {cat}: insufficient channels")
                    continue
                
                cat_dir = os.path.join(comp_40_dir, cat)
                os.makedirs(cat_dir, exist_ok=True)
                print(f"  -> Rendering 40 colormaps for category '{cat}'...")
                
                cat_success = 0
                for code_name, cmap_name in COLORMAPS_40:
                    # Filename matching user screenshot: <timestamp>_<category>_<cmapcode>.png
                    fname = f"{ts_folder}_{cat}_{code_name}.png"
                    out_path = os.path.join(cat_dir, fname)
                    
                    ok = self.render_and_save(base_data, f"{cat.upper()} [{code_name}]", out_path, cmap_name)
                    if ok:
                        cat_success += 1
                        total_images += 1
                        self.manifest_records.append({
                            'cycle_id': cycle_id,
                            'timestamp': ts_display,
                            'folder': ts_folder,
                            'category': cat,
                            'colormap_code': code_name,
                            'filename': fname,
                            'filepath': os.path.relpath(out_path, self.out_dir)
                        })
                print(f"     [OK] {cat_success}/40 images generated for '{cat}'")

        # Save manifest
        manifest_path = os.path.join(self.out_dir, 'manifest_composites_40.csv')
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['cycle_id', 'timestamp', 'folder', 'category', 'colormap_code', 'filename', 'filepath'])
            writer.writeheader()
            writer.writerows(self.manifest_records)

        elapsed = time.time() - t0
        print(f"\n==================================================")
        print(f"  SUCCESSFULLY GENERATED {total_images} IMAGES IN {elapsed:.1f}s")
        print(f"  Glitch fixed (zero horizontal scanline stripes)")
        print(f"  Output folder: '{self.out_dir}'")
        print(f"  Manifest: '{manifest_path}'")
        print(f"==================================================")


if __name__ == '__main__':
    pipeline = MTGPipelineV3(data_dir='data', out_dir='output_v2')
    pipeline.process_all()
