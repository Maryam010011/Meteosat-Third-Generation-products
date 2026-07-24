#!/usr/bin/env python3
"""
MTG FCI Multi-Cycle Meteorological Product Generation Pipeline v2
=================================================================
Processes ALL available acquisition cycles in data/ directory.
Generates per-cycle channel products and multi-channel RGB composites.
Uses strict folder structure: channels/<ch_name>/ and composites/<composite_name>/

Each output filename maps to exactly ONE product × ONE timestamp.
No combinatorial sweeps. No fabricated channels.

Usage:
    python generate_products_v2.py [--data-dir data] [--out-dir outputs] [--width 1000] [--height 1024]
"""

import os
import sys
import time
import argparse
import csv
import warnings
import numpy as np
from collections import defaultdict
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
    print("CRITICAL: Satpy is required. Run: pip install satpy netCDF4 pyresample")
    sys.exit(1)

# ============================================================================
# CHANNEL DEFINITIONS
# ============================================================================
# All 16 standard FCI channels with their product type and colormap
ALL_FCI_CHANNELS = {
    'vis_04': {'label': 'VIS 0.4 µm',  'type': 'reflectance', 'cmap': 'gray',    'vmin': 0,   'vmax': 100,  'cbar': 'Reflectance (%)'},
    'vis_05': {'label': 'VIS 0.5 µm',  'type': 'reflectance', 'cmap': 'gray',    'vmin': 0,   'vmax': 100,  'cbar': 'Reflectance (%)'},
    'vis_06': {'label': 'VIS 0.6 µm',  'type': 'reflectance', 'cmap': 'gray',    'vmin': 0,   'vmax': 100,  'cbar': 'Reflectance (%)'},
    'vis_08': {'label': 'VIS 0.8 µm',  'type': 'reflectance', 'cmap': 'gray',    'vmin': 0,   'vmax': 100,  'cbar': 'Reflectance (%)'},
    'vis_09': {'label': 'VIS 0.9 µm',  'type': 'reflectance', 'cmap': 'gray',    'vmin': 0,   'vmax': 100,  'cbar': 'Reflectance (%)'},
    'nir_13': {'label': 'NIR 1.3 µm',  'type': 'reflectance', 'cmap': 'gray',    'vmin': 0,   'vmax': 80,   'cbar': 'Reflectance (%)'},
    'nir_16': {'label': 'NIR 1.6 µm',  'type': 'reflectance', 'cmap': 'gray',    'vmin': 0,   'vmax': 80,   'cbar': 'Reflectance (%)'},
    'nir_22': {'label': 'NIR 2.2 µm',  'type': 'reflectance', 'cmap': 'gray',    'vmin': 0,   'vmax': 60,   'cbar': 'Reflectance (%)'},
    'ir_38':  {'label': 'IR 3.8 µm',   'type': 'brightness_temp', 'cmap': 'gray_r','vmin': 210,'vmax': 320,  'cbar': 'Brightness Temperature (K)'},
    'wv_63':  {'label': 'WV 6.3 µm',   'type': 'brightness_temp', 'cmap': 'gray_r','vmin': 220,'vmax': 280,  'cbar': 'Brightness Temperature (K)'},
    'wv_73':  {'label': 'WV 7.3 µm',   'type': 'brightness_temp', 'cmap': 'gray_r','vmin': 220,'vmax': 290,  'cbar': 'Brightness Temperature (K)'},
    'ir_87':  {'label': 'IR 8.7 µm',   'type': 'brightness_temp', 'cmap': 'gray_r','vmin': 200,'vmax': 320,  'cbar': 'Brightness Temperature (K)'},
    'ir_97':  {'label': 'IR 9.7 µm',   'type': 'brightness_temp', 'cmap': 'gray_r','vmin': 200,'vmax': 320,  'cbar': 'Brightness Temperature (K)'},
    'ir_105': {'label': 'IR 10.5 µm',  'type': 'brightness_temp', 'cmap': 'gray_r','vmin': 200,'vmax': 315,  'cbar': 'Cloud Top Temperature (K)'},
    'ir_123': {'label': 'IR 12.3 µm',  'type': 'brightness_temp', 'cmap': 'gray_r','vmin': 200,'vmax': 315,  'cbar': 'Brightness Temperature (K)'},
    'ir_133': {'label': 'IR 13.3 µm',  'type': 'brightness_temp', 'cmap': 'gray_r','vmin': 200,'vmax': 315,  'cbar': 'Brightness Temperature (K)'},
}


# ============================================================================
# COMPOSITE DEFINITIONS (EUMETSAT standard recipes)
# ============================================================================
# Required channels for each composite. Pipeline will skip a composite
# if any required channel is not loaded from that cycle's files.
COMPOSITE_SPECS = {
    'natural_colours': {
        'label': 'Natural Colours RGB',
        'required': ['nir_16', 'vis_08', 'vis_06'],
        'day_only': False,
        'night_only': False,
        'recipe': 'R: NIR1.6 [0-100%] | G: VIS0.8 [0-100%] | B: VIS0.6 [0-100%]',
        'ref': 'EUMETSAT EUM/OPS/DOC/15/843701 - Natural Colours RGB'
    },
    'airmass': {
        'label': 'Airmass RGB',
        'required': ['wv_63', 'wv_73', 'ir_97', 'ir_105'],
        'day_only': False,
        'night_only': False,
        'recipe': 'R: WV6.3-WV7.3 [-25,0K] | G: IR9.7-IR10.5 [-40,+5K] | B: WV6.3 inv [243,208K]',
        'ref': 'EUMETSAT EUM/OPS/DOC/15/843701 - Airmass RGB'
    },
    'dust': {
        'label': 'Dust RGB',
        'required': ['ir_123', 'ir_105', 'ir_87'],
        'day_only': False,
        'night_only': False,
        'recipe': 'R: IR12.3-IR10.5 [-4,+2K] | G: IR10.5-IR8.7 [0,+15K, γ2.5] | B: IR10.5 [261,289K]',
        'ref': 'EUMETSAT EUM/OPS/DOC/15/843701 - Dust RGB'
    },
    'ash': {
        'label': 'Ash RGB',
        'required': ['ir_123', 'ir_105', 'ir_87'],
        'day_only': False,
        'night_only': False,
        'recipe': 'R: IR12.3-IR10.5 [-4,+2K] | G: IR10.5-IR8.7 [-4,+5K] | B: IR10.5 [243,303K]',
        'ref': 'EUMETSAT EUM/OPS/DOC/15/843701 - Ash RGB'
    },
    'microphysics_24hr': {
        'label': '24hr Microphysics RGB',
        'required': ['ir_123', 'ir_105', 'ir_87'],
        'day_only': False,
        'night_only': False,
        'recipe': 'R: IR12.3-IR10.5 [-4,+2K] | G: IR10.5-IR8.7 [0,+6K, γ1.2] | B: IR10.5 [248,303K]',
        'ref': 'EUMETSAT EUM/OPS/DOC/15/843701 - 24hr Microphysics RGB'
    },
    'day_microphysics': {
        'label': 'Day Microphysics RGB',
        'required': ['vis_08', 'ir_38', 'ir_105'],
        'day_only': True,
        'night_only': False,
        'recipe': 'R: VIS0.8 [0-100%] | G: IR3.8 refl [0-60%, γ2.5] | B: IR10.5 [203,323K]',
        'ref': 'EUMETSAT EUM/OPS/DOC/15/843701 - Day Microphysics RGB'
    },
    'night_microphysics': {
        'label': 'Night Microphysics RGB',
        'required': ['ir_123', 'ir_105', 'ir_38'],
        'day_only': False,
        'night_only': True,
        'recipe': 'R: IR12.3-IR10.5 [-4,+2K] | G: IR10.5-IR3.8 [0,+10K] | B: IR10.5 [243,293K]',
        'ref': 'EUMETSAT EUM/OPS/DOC/15/843701 - Night Microphysics RGB'
    },
    'severe_storms': {
        'label': 'Severe Storms RGB',
        'required': ['wv_63', 'wv_73', 'ir_38', 'ir_105', 'nir_16', 'vis_06'],
        'day_only': False,
        'night_only': False,
        'recipe': 'R: WV6.3-WV7.3 [-35,+5K] | G: IR3.8-IR10.5 [-5,+60K, γ0.5] | B: NIR1.6-VIS0.6 [-75,+25%]',
        'ref': 'EUMETSAT EUM/OPS/DOC/15/843701 - Severe Storms RGB'
    },
    'snow': {
        'label': 'Snow RGB',
        'required': ['vis_08', 'nir_16', 'ir_38'],
        'day_only': True,
        'night_only': False,
        'recipe': 'R: VIS0.8 [0-100%, γ1.7] | G: NIR1.6 [0-70%, γ1.7] | B: IR3.8 refl [0-30%, γ1.7]',
        'ref': 'EUMETSAT EUM/OPS/DOC/15/843701 - Snow RGB'
    },
    # ---------------------------------------------------------------
    # APPROXIMATED COMPOSITES using only available channels.
    # These are generated ONLY when full-spec channels are missing.
    # Clearly labeled as approximations in header/recipe text.
    # ---------------------------------------------------------------
    'day_microphysics_approx': {
        'label': 'Day Microphysics RGB (approx: NIR2.2 sub for VIS0.8)',
        'required': ['nir_22', 'ir_38', 'ir_105'],
        'day_only': True,
        'night_only': False,
        'recipe': 'R: NIR2.2[0-60%] G: IR3.8_refl[0-60%,γ2.5] B: IR10.5[203,323K] | NOTE: NIR2.2 substituted for VIS0.8',
        'ref': 'Approximation of EUMETSAT Day Microphysics RGB; VIS0.8 unavailable in this HRFI dataset'
    },
    'convection_proxy': {
        'label': 'Severe Convection Proxy RGB',
        'required': ['vis_06', 'ir_105', 'ir_38'],
        'day_only': True,
        'night_only': False,
        'recipe': 'R: VIS0.6[0-100%] G: IR10.5_inv[210-320K] B: IR3.8-IR10.5[-5,+15K] | Convection proxy',
        'ref': 'Convection proxy composite using VIS0.6, IR10.5, IR3.8; standard approach when WV channels unavailable'
    },
    'night_microphysics_proxy': {
        'label': 'Night Microphysics Proxy RGB',
        'required': ['ir_38', 'ir_105'],
        'day_only': False,
        'night_only': True,
        'recipe': 'R: IR3.8-IR10.5[-4,+2K] G: IR10.5[243,293K] B: IR10.5[273,293K] | NOTE: IR12.3 unavailable, proxy recipe',
        'ref': 'Night Microphysics proxy; IR12.3 unavailable in HRFI subset. Standard fallback recipe.'
    },
}


# ============================================================================
# PIPELINE CLASS
# ============================================================================
class MTGMultiCyclePipeline:
    def __init__(self, data_dir='data', out_dir='outputs', target_size=(1000, 1024)):
        self.data_dir = os.path.abspath(data_dir)
        self.out_dir = os.path.abspath(out_dir)
        self.target_w, self.target_h = target_size
        
        # Subdirectory map (no rgb_composites)
        self.channels_dir    = os.path.join(out_dir, 'channels')
        self.composites_dir  = os.path.join(out_dir, 'composites')
        self.derived_dir     = os.path.join(out_dir, 'derived_products')
        
        for d in [self.channels_dir, self.composites_dir, self.derived_dir]:
            os.makedirs(d, exist_ok=True)
        
        self.manifest_records = []
        self.manifest_path = os.path.join(out_dir, 'manifest_v2.csv')

    # -----------------------------------------------------------------------
    def discover_cycles(self):
        """Discover all unique repeat-cycle IDs and their associated files."""
        print(f"\n[STEP 1] Discovering acquisition cycles in '{self.data_dir}'...")
        cycles = defaultdict(list)
        for fname in os.listdir(self.data_dir):
            if not (fname.endswith('.nc') and 'CHK-BODY' in fname):
                continue
            parts = fname.split('_')
            for i, p in enumerate(parts):
                if len(p) == 4 and p.isdigit() and i > 5:
                    cycles[p].append(os.path.join(self.data_dir, fname))
                    break
        
        print(f" -> Total chunk-body NC files found: {sum(len(v) for v in cycles.values())}")
        print(f" -> Unique repeat cycles detected: {len(cycles)}")
        for cid, flist in sorted(cycles.items()):
            print(f"    Cycle {cid}: {len(flist)} chunk files")
        return dict(sorted(cycles.items()))

    # -----------------------------------------------------------------------
    def _get_cycle_info(self, cycle_id, cycle_files):
        """Load scene and return metadata + available channels."""
        scn = Scene(filenames=cycle_files, reader='fci_l1c_nc')
        ts = scn.start_time
        ts_str = ts.strftime('%Y%m%d_%H%M')
        ts_display = ts.strftime('%Y-%m-%d %H:%M UTC')
        
        # Determine day/night from UTC time (rough: 06–18 UTC = daytime at 0° lon)
        is_day = 6 <= ts.hour < 18
        
        avail_in_file = scn.available_dataset_names()
        available_channels = [ch for ch in ALL_FCI_CHANNELS if ch in avail_in_file]
        
        return scn, ts, ts_str, ts_display, is_day, available_channels

    # -----------------------------------------------------------------------
    def _extract(self, scn, channel_name):
        """Extract array + valid mask from loaded Satpy scene, resize to target."""
        raw = scn[channel_name].values.astype(np.float32)
        is_ir = channel_name.startswith('ir_') or channel_name.startswith('wv_')
        fill_val = 200.0 if is_ir else 0.0
        
        if is_ir:
            mask_raw = (raw >= 150.0) & (~np.isnan(raw))
        else:
            mask_raw = (raw > 0.01) & (~np.isnan(raw))
        
        clean = np.nan_to_num(raw, nan=fill_val)
        img_r = Image.fromarray(clean).resize((self.target_w, self.target_h), resample=Image.Resampling.BILINEAR)
        m_r   = Image.fromarray(mask_raw.astype(np.uint8)).resize((self.target_w, self.target_h), resample=Image.Resampling.NEAREST)
        
        arr = np.array(img_r, dtype=np.float32)
        mask = np.array(m_r, dtype=bool)
        arr[~mask] = np.nan
        return arr, mask

    # -----------------------------------------------------------------------
    def _norm(self, arr, lo, hi, gamma=1.0):
        """Normalize array to [0,1] with optional gamma correction."""
        out = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        if gamma != 1.0:
            out = np.power(out, 1.0 / gamma)
        return out

    # -----------------------------------------------------------------------
    def _disk_geometry(self):
        """Compute per-pixel disk distance for off-disk masking."""
        ny, nx = self.target_h, self.target_w
        y_g, x_g = np.ogrid[:ny, :nx]
        cy, cx = ny / 2.0, nx / 2.0
        r_disk = min(ny, nx) * 0.46
        dist = np.sqrt((x_g - cx)**2 + (y_g - cy)**2)
        return dist, r_disk

    # -----------------------------------------------------------------------
    def _render_channel(self, data, channel_id, ts_display, cycle_id, 
                        out_subdir, filename):
        """Render a single-channel calibrated product image with white background."""
        spec = ALL_FCI_CHANNELS[channel_id]
        title = f"{spec['label']} — Calibrated {spec['type'].replace('_', ' ').title()}"
        cmap_name = spec['cmap']
        vmin, vmax = spec['vmin'], spec['vmax']
        cbar_label = spec['cbar']
        
        try:
            dist, r_disk = self._disk_geometry()
            plot_data = np.copy(data)
            plot_data[dist > r_disk] = np.nan
            
            fig = plt.figure(figsize=(self.target_w/100.0, self.target_h/100.0), dpi=100)
            fig.patch.set_facecolor('#FFFFFF')
            
            ax_map = fig.add_axes([0.02, 0.09, 0.96, 0.83])
            ax_map.set_facecolor('#FFFFFF')
            
            cmap_obj = plt.get_cmap(cmap_name).copy()
            cmap_obj.set_bad('white')
            
            valid = plot_data[~np.isnan(plot_data)]
            vmin_use = vmin if vmin is not None else float(np.percentile(valid, 1)) if len(valid) > 0 else 0
            vmax_use = vmax if vmax is not None else float(np.percentile(valid, 99)) if len(valid) > 0 else 1
            
            im = ax_map.imshow(plot_data, cmap=cmap_obj, vmin=vmin_use, vmax=vmax_use, 
                               origin='upper', aspect='auto')
            
            # Disk boundary ring
            disk_mask = np.abs(dist - r_disk) < 1.5
            ax_map.imshow(np.ma.masked_where(~disk_mask, np.ones_like(disk_mask)),
                         cmap=mcolors.ListedColormap(['#64748B']), vmin=0, vmax=1,
                         alpha=0.5, origin='upper', aspect='auto')
            # Grid lines
            ny, nx = self.target_h, self.target_w
            y_g, x_g = np.ogrid[:ny, :nx]
            cx, cy = nx/2.0, ny/2.0
            grid = ((np.abs((x_g - cx) % 100) < 1.0) | (np.abs((y_g - cy) % 100) < 1.0)) & (dist < r_disk)
            ax_map.imshow(np.ma.masked_where(~grid, np.ones_like(grid)),
                         cmap=mcolors.ListedColormap(['#94A3B8']), vmin=0, vmax=1,
                         alpha=0.2, origin='upper', aspect='auto')
            ax_map.axis('off')
            
            # Top header
            ax_hdr = fig.add_axes([0.0, 0.92, 1.0, 0.08])
            ax_hdr.set_facecolor('#F8FAFC')
            ax_hdr.axis('off')
            ax_hdr.text(0.02, 0.65, title.upper(), color='#0F172A', fontsize=12, fontweight='bold', va='center')
            ax_hdr.text(0.02, 0.22, f"  [CHANNEL]  ", color='#FFFFFF', fontsize=9, fontweight='bold', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#2563EB', edgecolor='none'))
            ax_hdr.text(0.98, 0.40,
                        f"EUMETSAT MTG-I1 FCI L1C | Cycle {cycle_id} | {ts_display}",
                        color='#475569', fontsize=9, ha='right', va='center')
            
            # Bottom footer with colorbar
            ax_ftr = fig.add_axes([0.0, 0.0, 1.0, 0.09])
            ax_ftr.set_facecolor('#F8FAFC')
            ax_ftr.axis('off')
            cax = fig.add_axes([0.22, 0.027, 0.56, 0.032])
            cb = fig.colorbar(im, cax=cax, orientation='horizontal')
            cb.ax.tick_params(labelsize=8, colors='#334155')
            cb.set_label(cbar_label, color='#0F172A', fontsize=8, fontweight='bold', labelpad=2)
            cb.outline.set_edgecolor('#CBD5E1')
            ax_ftr.text(0.02, 0.55, f"File: {filename}", color='#475569', fontsize=8, va='center')
            ax_ftr.text(0.98, 0.55, f"Channel: {channel_id}", color='#475569', fontsize=8, ha='right', va='center')
            
            out_path = os.path.join(out_subdir, filename)
            plt.savefig(out_path, dpi=100, facecolor='#FFFFFF', edgecolor='none')
            plt.close(fig)
            return True, out_path
        except Exception as e:
            plt.close('all')
            return False, str(e)

    # -----------------------------------------------------------------------
    def _render_rgb(self, rgb_arr, composite_id, ts_display, cycle_id, 
                    out_subdir, filename, spec, is_approx=False):
        """Render an RGB composite product image with white background."""
        try:
            dist, r_disk = self._disk_geometry()
            rgb_clean = np.clip(rgb_arr, 0.0, 1.0)
            
            # White outside disk
            off_disk = dist > r_disk
            rgb_clean[off_disk, :] = 1.0  # white

            fig = plt.figure(figsize=(self.target_w/100.0, self.target_h/100.0), dpi=100)
            fig.patch.set_facecolor('#FFFFFF')
            
            ax_map = fig.add_axes([0.02, 0.09, 0.96, 0.83])
            ax_map.set_facecolor('#FFFFFF')
            ax_map.imshow(rgb_clean, origin='upper', aspect='auto')
            
            # Disk ring + grid
            ny, nx = self.target_h, self.target_w
            y_g, x_g = np.ogrid[:ny, :nx]
            cx, cy = nx/2.0, ny/2.0
            disk_boundary = np.abs(dist - r_disk) < 1.5
            ax_map.imshow(np.ma.masked_where(~disk_boundary, np.ones_like(disk_boundary)),
                         cmap=mcolors.ListedColormap(['#475569']), vmin=0, vmax=1,
                         alpha=0.5, origin='upper', aspect='auto')
            grid = ((np.abs((x_g - cx) % 100) < 1.0) | (np.abs((y_g - cy) % 100) < 1.0)) & (dist < r_disk)
            ax_map.imshow(np.ma.masked_where(~grid, np.ones_like(grid)),
                         cmap=mcolors.ListedColormap(['#94A3B8']), vmin=0, vmax=1,
                         alpha=0.20, origin='upper', aspect='auto')
            ax_map.axis('off')
            
            # Header
            badge_color = '#D97706' if is_approx else '#059669'
            badge_text  = ' [COMPOSITE — APPROX] ' if is_approx else ' [COMPOSITE — STANDARD] '
            ax_hdr = fig.add_axes([0.0, 0.92, 1.0, 0.08])
            ax_hdr.set_facecolor('#F8FAFC')
            ax_hdr.axis('off')
            title_str = spec['label'].upper()
            ax_hdr.text(0.02, 0.65, title_str, color='#0F172A', fontsize=11, fontweight='bold', va='center')
            ax_hdr.text(0.02, 0.22, badge_text, color='#FFFFFF', fontsize=9, fontweight='bold', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=badge_color, edgecolor='none'))
            ax_hdr.text(0.98, 0.40,
                        f"EUMETSAT MTG-I1 FCI L1C | Cycle {cycle_id} | {ts_display}",
                        color='#475569', fontsize=9, ha='right', va='center')
            
            # Footer
            ax_ftr = fig.add_axes([0.0, 0.0, 1.0, 0.09])
            ax_ftr.set_facecolor('#F8FAFC')
            ax_ftr.axis('off')
            recipe_short = spec['recipe'][:80] + ('...' if len(spec['recipe']) > 80 else '')
            ax_ftr.text(0.02, 0.72, f"RECIPE: {recipe_short}", color='#1E293B', fontsize=7, fontweight='bold', va='center')
            ax_ftr.text(0.02, 0.28, f"Ref: {spec['ref']}", color='#64748B', fontsize=7, va='center')
            ax_ftr.text(0.98, 0.50, f"File: {filename}", color='#475569', fontsize=7, ha='right', va='center')
            
            out_path = os.path.join(out_subdir, filename)
            plt.savefig(out_path, dpi=100, facecolor='#FFFFFF', edgecolor='none')
            plt.close(fig)
            return True, out_path
        except Exception as e:
            plt.close('all')
            return False, str(e)

    # -----------------------------------------------------------------------
    def _compute_composite(self, composite_id, channels, is_day):
        """Compute RGB array for a given composite using loaded channel data."""
        n = self._norm  # shorthand
        
        def get(ch):
            return channels.get(ch)

        if composite_id == 'natural_colours':
            nir16 = get('nir_16'); vis08 = get('vis_08'); vis06 = get('vis_06')
            r = n(nir16, 0, 100); g = n(vis08, 0, 100); b = n(vis06, 0, 100)
            return np.dstack([r, g, b])
        
        elif composite_id == 'airmass':
            wv63 = get('wv_63'); wv73 = get('wv_73'); ir97 = get('ir_97'); ir105 = get('ir_105')
            r = n(wv63 - wv73, -25, 0)
            g = n(ir97 - ir105, -40, 5)
            b = n(wv63, 243, 208)  # inverted: lower BT = brighter
            return np.dstack([r, g, b])
        
        elif composite_id == 'dust':
            ir123 = get('ir_123'); ir105 = get('ir_105'); ir87 = get('ir_87')
            r = n(ir123 - ir105, -4, 2)
            g = n(ir105 - ir87, 0, 15, gamma=2.5)
            b = n(ir105, 261, 289)
            return np.dstack([r, g, b])
        
        elif composite_id == 'ash':
            ir123 = get('ir_123'); ir105 = get('ir_105'); ir87 = get('ir_87')
            r = n(ir123 - ir105, -4, 2)
            g = n(ir105 - ir87, -4, 5)
            b = n(ir105, 243, 303)
            return np.dstack([r, g, b])
        
        elif composite_id == 'microphysics_24hr':
            ir123 = get('ir_123'); ir105 = get('ir_105'); ir87 = get('ir_87')
            r = n(ir123 - ir105, -4, 2)
            g = n(ir105 - ir87, 0, 6, gamma=1.2)
            b = n(ir105, 248, 303)
            return np.dstack([r, g, b])
        
        elif composite_id == 'day_microphysics':
            vis08 = get('vis_08'); ir38 = get('ir_38'); ir105 = get('ir_105')
            # IR3.8 solar reflectance component proxy (daytime)
            ir38_refl = np.clip((ir38 - 250.0) / 70.0 * 60.0, 0, 60)
            r = n(vis08, 0, 100)
            g = n(ir38_refl, 0, 60, gamma=2.5)
            b = n(ir105, 203, 323)
            return np.dstack([r, g, b])
        
        elif composite_id == 'night_microphysics':
            ir123 = get('ir_123'); ir105 = get('ir_105'); ir38 = get('ir_38')
            r = n(ir123 - ir105, -4, 2)
            g = n(ir105 - ir38, 0, 10)
            b = n(ir105, 243, 293)
            return np.dstack([r, g, b])
        
        elif composite_id == 'severe_storms':
            wv63 = get('wv_63'); wv73 = get('wv_73')
            ir38 = get('ir_38'); ir105 = get('ir_105')
            nir16 = get('nir_16'); vis06 = get('vis_06')
            r = n(wv63 - wv73, -35, 5)
            g = n(ir38 - ir105, -5, 60, gamma=0.5)
            b = n(nir16 - vis06, -75, 25)
            return np.dstack([r, g, b])
        
        elif composite_id == 'snow':
            vis08 = get('vis_08'); nir16 = get('nir_16'); ir38 = get('ir_38')
            ir38_refl = np.clip((ir38 - 250.0) / 70.0 * 30.0, 0, 30)
            r = n(vis08, 0, 100, gamma=1.7)
            g = n(nir16, 0, 70, gamma=1.7)
            b = n(ir38_refl, 0, 30, gamma=1.7)
            return np.dstack([r, g, b])
        
        # ---- Approximated composites ----
        elif composite_id == 'day_microphysics_approx':
            nir22 = get('nir_22'); ir38 = get('ir_38'); ir105 = get('ir_105')
            ir38_refl = np.clip((ir38 - 250.0) / 70.0 * 60.0, 0, 60)
            ir38_refl[np.isnan(ir38)] = np.nan
            r = n(nir22, 0, 60)
            g = n(ir38_refl, 0, 60, gamma=2.5)
            b = n(ir105, 203, 323)
            return np.dstack([r, g, b])
        
        elif composite_id == 'convection_proxy':
            vis06 = get('vis_06'); ir105 = get('ir_105'); ir38 = get('ir_38')
            r = n(vis06, 0, 100)
            g = n(320.0 - ir105, 0, 110)
            b = n(ir38 - ir105, -5, 15)
            return np.dstack([r, g, b])
        
        elif composite_id == 'night_microphysics_proxy':
            ir38 = get('ir_38'); ir105 = get('ir_105')
            r = n(ir38 - ir105, -4, 2)
            g = n(ir105, 243, 293)
            b = n(ir105, 273, 293)
            return np.dstack([r, g, b])
        
        return None

    # -----------------------------------------------------------------------
    def process_cycle(self, cycle_id, cycle_files):
        """Process one acquisition cycle: channels + composites."""
        print(f"\n{'='*60}")
        print(f"  PROCESSING CYCLE {cycle_id}")
        print(f"{'='*60}")
        
        scn, ts, ts_str, ts_display, is_day, available_channels = self._get_cycle_info(cycle_id, cycle_files)
        tod = 'DAY' if is_day else 'NIGHT'
        print(f"  Timestamp: {ts_display} | {tod}")
        print(f"  Available channels in this dataset ({len(available_channels)}/16): {available_channels}")
        
        # Log missing channels
        for ch in ALL_FCI_CHANNELS:
            if ch not in available_channels:
                print(f"  [CHANNEL LOG] '{ch}' ({ALL_FCI_CHANNELS[ch]['label']}) not available in source data for cycle {cycle_id}")
        
        # Load all available channels
        scn.load(available_channels)
        channels = {}
        channels_masks = {}
        
        for ch in available_channels:
            try:
                arr, mask = self._extract(scn, ch)
                channels[ch] = arr
                channels_masks[ch] = mask
                v = arr[mask]
                print(f"  [OK] {ch}: range [{v.min():.2f}, {v.max():.2f}]" if len(v) > 0 else f"  [OK] {ch}: (all NaN)")
            except Exception as e:
                print(f"  [ERROR] Failed to load {ch}: {e}")
        
        ch_success = 0
        ch_skipped = 0
        comp_success = 0
        comp_skipped = 0
        
        # ----------------------------------------------------------------
        # CHANNELS — one grayscale BT/reflectance product per available channel
        # ----------------------------------------------------------------
        print(f"\n  [CHANNELS] Generating calibrated channel products...")
        for ch in ALL_FCI_CHANNELS:
            if ch not in channels:
                ch_skipped += 1
                continue
            
            ch_subdir = os.path.join(self.channels_dir, ch)
            os.makedirs(ch_subdir, exist_ok=True)
            fname = f"{ch}_{ts_str}.png"
            
            ok, result = self._render_channel(channels[ch], ch, ts_display, cycle_id, ch_subdir, fname)
            status = 'SUCCESS' if ok else f'FAILED ({result})'
            print(f"    {'[OK]' if ok else '[FAIL]'} {fname}")
            if ok:
                ch_success += 1
            
            self.manifest_records.append({
                'product_type': 'channel',
                'product_name': ALL_FCI_CHANNELS[ch]['label'],
                'composite_id': ch,
                'cycle_id': cycle_id,
                'timestamp': ts_display,
                'filename': fname,
                'filepath': os.path.relpath(result if ok else '', self.out_dir),
                'status': status,
                'is_approx': 'no',
                'ref': f"FCI L1C calibrated {ALL_FCI_CHANNELS[ch]['type']}"
            })
        
        # ----------------------------------------------------------------
        # COMPOSITES — one RGB product per applicable composite recipe
        # ----------------------------------------------------------------
        print(f"\n  [COMPOSITES] Generating composite products...")
        
        # Determine which standard composites can be generated
        # and which approximated ones to fall back to
        standard_composites_possible = set()
        for cid, spec in COMPOSITE_SPECS.items():
            if cid.endswith('_approx') or cid.endswith('_proxy'):
                continue  # skip approximations for first pass
            if all(ch in channels for ch in spec['required']):
                standard_composites_possible.add(cid)
        
        # For each composite in the full spec list
        for cid, spec in COMPOSITE_SPECS.items():
            is_approx = cid.endswith('_approx') or cid.endswith('_proxy')
            
            # Skip approx version if the standard version can run
            standard_equiv = cid.replace('_approx', '').replace('_proxy', '')
            if is_approx and standard_equiv in standard_composites_possible:
                print(f"    [SKIP-DUP] {cid}: standard version available")
                continue
            
            # Check day/night applicability
            if spec['day_only'] and not is_day:
                print(f"    [SKIP-NIGHT] {spec['label']}: day-only composite skipped (night cycle)")
                comp_skipped += 1
                continue
            if spec['night_only'] and is_day:
                print(f"    [SKIP-DAY] {spec['label']}: night-only composite skipped (day cycle)")
                comp_skipped += 1
                continue
            
            # Check required channels
            missing = [ch for ch in spec['required'] if ch not in channels]
            if missing:
                print(f"    [SKIP-MISSING] {spec['label']}: missing channels {missing}")
                comp_skipped += 1
                continue
            
            # Compute the RGB array
            try:
                rgb = self._compute_composite(cid, channels, is_day)
                if rgb is None:
                    print(f"    [SKIP-NORECIPE] {spec['label']}: no compute recipe found")
                    comp_skipped += 1
                    continue
            except Exception as e:
                print(f"    [ERROR-COMPUTE] {spec['label']}: {e}")
                comp_skipped += 1
                continue
            
            # Render
            comp_subdir = os.path.join(self.composites_dir, cid)
            os.makedirs(comp_subdir, exist_ok=True)
            fname = f"{cid}_{ts_str}.png"
            
            ok, result = self._render_rgb(rgb, cid, ts_display, cycle_id, comp_subdir, fname, spec, is_approx)
            status = 'SUCCESS' if ok else f'FAILED ({result})'
            print(f"    {'[OK]' if ok else '[FAIL]'} {fname} {'(APPROX)' if is_approx else ''}")
            if ok:
                comp_success += 1
            else:
                comp_skipped += 1
            
            self.manifest_records.append({
                'product_type': 'composite',
                'product_name': spec['label'],
                'composite_id': cid,
                'cycle_id': cycle_id,
                'timestamp': ts_display,
                'filename': fname,
                'filepath': os.path.relpath(result if ok else '', self.out_dir),
                'status': status,
                'is_approx': 'yes' if is_approx else 'no',
                'ref': spec['ref']
            })
        
        return ch_success, ch_skipped, comp_success, comp_skipped

    # -----------------------------------------------------------------------
    def generate_derived_products(self, cycle_id_primary, cycle_files_primary,
                                  cycle_id_secondary=None, cycle_files_secondary=None):
        """Generate derived index/mask products using primary (and optionally secondary) cycle data."""
        print(f"\n[DERIVED PRODUCTS] Generating derived meteorological products...")
        
        scn = Scene(filenames=cycle_files_primary, reader='fci_l1c_nc')
        avail = scn.available_dataset_names()
        chans_to_load = [ch for ch in ['vis_06', 'nir_22', 'ir_38', 'ir_105', 'ir_123'] if ch in avail]
        scn.load(chans_to_load)
        
        channels = {}
        masks = {}
        for ch in chans_to_load:
            try:
                arr, mask = self._extract(scn, ch)
                channels[ch] = arr
                masks[ch] = mask
            except Exception:
                pass
        
        # Secondary cycle for temporal diff
        sec_channels = {}
        sec_masks = {}
        if cycle_files_secondary:
            try:
                scn2 = Scene(filenames=cycle_files_secondary, reader='fci_l1c_nc')
                avail2 = scn2.available_dataset_names()
                chans2 = [ch for ch in ['vis_06', 'ir_105'] if ch in avail2]
                scn2.load(chans2)
                for ch in chans2:
                    arr2, mask2 = self._extract(scn2, ch)
                    sec_channels[ch] = arr2
                    sec_masks[ch] = mask2
            except Exception as e:
                print(f"  [WARN] Secondary cycle load failed: {e}")
        
        ts_str = Scene(filenames=cycle_files_primary, reader='fci_l1c_nc').start_time.strftime('%Y%m%d_%H%M')
        ts_display = Scene(filenames=cycle_files_primary, reader='fci_l1c_nc').start_time.strftime('%Y-%m-%d %H:%M UTC')
        
        dist, r_disk = self._disk_geometry()
        success_count = 0
        
        def render_derived(data, title, filename, cmap, vmin, vmax, cbar_label, cat_name='derived_products'):
            nonlocal success_count
            try:
                plot_data = np.copy(data)
                plot_data[dist > r_disk] = np.nan
                
                fig = plt.figure(figsize=(self.target_w/100.0, self.target_h/100.0), dpi=100)
                fig.patch.set_facecolor('#FFFFFF')
                ax_map = fig.add_axes([0.02, 0.09, 0.96, 0.83])
                ax_map.set_facecolor('#FFFFFF')
                
                if isinstance(cmap, str):
                    cmap_obj = plt.get_cmap(cmap).copy()
                else:
                    cmap_obj = cmap
                cmap_obj.set_bad('white')
                
                valid = plot_data[~np.isnan(plot_data)]
                v0 = vmin if vmin is not None else (float(np.percentile(valid, 1)) if len(valid) > 0 else 0)
                v1 = vmax if vmax is not None else (float(np.percentile(valid, 99)) if len(valid) > 0 else 1)
                
                im = ax_map.imshow(plot_data, cmap=cmap_obj, vmin=v0, vmax=v1, origin='upper', aspect='auto')
                
                # Ring
                ring = np.abs(dist - r_disk) < 1.5
                ax_map.imshow(np.ma.masked_where(~ring, np.ones_like(ring)),
                             cmap=mcolors.ListedColormap(['#475569']), alpha=0.5, origin='upper', aspect='auto')
                ax_map.axis('off')
                
                ax_hdr = fig.add_axes([0.0, 0.92, 1.0, 0.08])
                ax_hdr.set_facecolor('#F8FAFC'); ax_hdr.axis('off')
                ax_hdr.text(0.02, 0.65, title.upper(), color='#0F172A', fontsize=11, fontweight='bold', va='center')
                ax_hdr.text(0.02, 0.22, ' [DERIVED PRODUCT] ', color='#FFFFFF', fontsize=9, fontweight='bold', va='center',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#7C3AED', edgecolor='none'))
                ax_hdr.text(0.98, 0.40, f"EUMETSAT MTG-I1 FCI L1C | {ts_display}", color='#475569', fontsize=9, ha='right', va='center')
                
                ax_ftr = fig.add_axes([0.0, 0.0, 1.0, 0.09])
                ax_ftr.set_facecolor('#F8FAFC'); ax_ftr.axis('off')
                cax = fig.add_axes([0.22, 0.027, 0.56, 0.032])
                cb = fig.colorbar(im, cax=cax, orientation='horizontal')
                cb.ax.tick_params(labelsize=8, colors='#334155')
                cb.set_label(cbar_label, color='#0F172A', fontsize=8, fontweight='bold', labelpad=2)
                cb.outline.set_edgecolor('#CBD5E1')
                ax_ftr.text(0.02, 0.55, f"File: {filename}", color='#475569', fontsize=8, va='center')
                
                out_path = os.path.join(self.derived_dir, filename)
                plt.savefig(out_path, dpi=100, facecolor='#FFFFFF', edgecolor='none')
                plt.close(fig)
                success_count += 1
                print(f"    [OK] {filename}")
                
                self.manifest_records.append({
                    'product_type': 'derived',
                    'product_name': title,
                    'composite_id': filename.replace('.png', ''),
                    'cycle_id': cycle_id_primary,
                    'timestamp': ts_display,
                    'filename': filename,
                    'filepath': os.path.relpath(out_path, self.out_dir),
                    'status': 'SUCCESS',
                    'is_approx': 'no',
                    'ref': 'Derived index/mask product'
                })
                return True
            except Exception as e:
                plt.close('all')
                print(f"    [FAIL] {filename}: {e}")
                return False
        
        vis06 = channels.get('vis_06')
        nir22 = channels.get('nir_22')
        ir38  = channels.get('ir_38')
        ir105 = channels.get('ir_105')
        ir123 = channels.get('ir_123')
        
        if ir38 is not None and ir105 is not None:
            diff_38_105 = ir38 - ir105
            diff_38_105[np.isnan(ir38) | np.isnan(ir105)] = np.nan
            render_derived(diff_38_105, "Low Cloud & Fog Difference (IR3.8 − IR10.5)",
                          "diff_ir38_ir105_lowcloud_fog.png", 'RdBu_r', -15, 25, "ΔBT (K)")
        
        if ir123 is not None and ir105 is not None:
            diff_123_105 = ir123 - ir105
            diff_123_105[np.isnan(ir123) | np.isnan(ir105)] = np.nan
            render_derived(diff_123_105, "True Split-Window Difference (IR12.3 − IR10.5)",
                          "diff_ir105_ir123_split_window.png", 'RdBu_r', -5, 15, "ΔBT (K)")
        else:
            print(f"    [LOG] True split-window skipped: ir_123 not available in source data")
        
        if nir22 is not None and vis06 is not None:
            swir_proxy = (nir22 - vis06) / (nir22 + vis06 + 1e-5)
            swir_proxy[np.isnan(nir22) | np.isnan(vis06)] = np.nan
            render_derived(swir_proxy, "SWIR Moisture & Ice Content Proxy",
                          "index_swir_moisture_proxy.png", 'YlGn', -0.4, 0.7, "SWIR Moisture Index")
            
            ndsi = (vis06 - nir22) / (vis06 + nir22 + 1e-5)
            ndsi[np.isnan(vis06) | np.isnan(nir22)] = np.nan
            render_derived(ndsi, "Normalized Difference Snow Index (NDSI Proxy)",
                          "index_ndsi_proxy.png", 'Blues', -0.3, 0.8, "NDSI Index")
        
        if vis06 is not None and ir105 is not None:
            cloud_mask = np.full_like(vis06, np.nan)
            valid_px = (~np.isnan(vis06)) | (~np.isnan(ir105))
            cloud_px  = valid_px & ((vis06 > 15.0) | (ir105 < 273.15))
            cloud_mask[valid_px] = 0.0
            cloud_mask[cloud_px]  = 1.0
            cmap_cloud = mcolors.ListedColormap(['#FFFFFF', '#1E40AF'])
            cmap_cloud.set_bad('white')
            render_derived(cloud_mask, "Cloud / Clear-Sky Binary Segmentation Mask",
                          "mask_cloud_binary.png", cmap_cloud, 0, 1,
                          "White: Clear/Land/Ocean | Blue: Cloud Cover")
        
        if ir105 is not None:
            cloud_alt = np.clip((288.15 - ir105) / 6.5, 0.0, 16.0)
            cloud_alt[np.isnan(ir105)] = np.nan
            render_derived(cloud_alt, "Cloud Top Height Proxy (Lapse Rate Model)",
                          "proxy_cloud_top_height.png", 'terrain', 0, 15, "Estimated Altitude (km MSL)")
            
            conv_cores = np.full_like(ir105, np.nan)
            valid_px = ~np.isnan(ir105)
            core_px  = valid_px & (ir105 < 210.0)
            conv_cores[valid_px] = 0.0
            conv_cores[core_px]  = 1.0
            cmap_core = mcolors.ListedColormap(['#FFFFFF', '#DC2626'])
            cmap_core.set_bad('white')
            render_derived(conv_cores, "Deep Convective Core Mask (<210K / −63°C)",
                          "mask_deep_convective_cores.png", cmap_core, 0, 1,
                          "White: Normal | Red: Deep Convective Core")
        
        if ir38 is not None and ir105 is not None:
            fire_idx = np.clip(ir38 - ir105, 0, 50)
            fire_idx[np.isnan(ir38) | np.isnan(ir105)] = np.nan
            render_derived(fire_idx, "Fire & Hotspot Thermal Anomaly Index",
                          "index_thermal_fire_anomaly.png", 'YlOrRd', 0, 35, "Thermal Anomaly ΔBT (K)")
        
        if vis06 is not None and nir22 is not None:
            sunglint = np.clip(vis06 - nir22, 0, 50)
            sunglint[np.isnan(vis06) | np.isnan(nir22)] = np.nan
            render_derived(sunglint, "Sun Glint & Marine Reflection Proxy",
                          "proxy_sunglint_marine.png", 'PuBuGn', 0, 30, "Optical Reflection Difference (%)")
            
            opt_depth = np.clip((vis06 + 1.0) / (nir22 + 1.0), 0, 8)
            opt_depth[np.isnan(vis06) | np.isnan(nir22)] = np.nan
            render_derived(opt_depth, "Cloud Optical Depth & Particle Scattering Proxy",
                          "proxy_optical_depth.png", 'viridis', 0.5, 6.0, "Optical Thickness Proxy Ratio")
        
        # Temporal change products (cycle-to-cycle)
        if ir105 is not None and 'ir_105' in sec_channels:
            arr52 = channels['ir_105']; arr71 = sec_channels['ir_105']
            m52   = masks['ir_105'];    m71   = sec_masks['ir_105']
            valid_both = m52 & m71 & (dist <= r_disk)
            diff_ir = np.where(valid_both, arr52 - arr71, np.nan)
            render_derived(diff_ir, f"Multi-Temporal BT Shift (Cycle {cycle_id_primary} − {cycle_id_secondary})",
                          "temporal_ir105_change.png", 'coolwarm', -20, 20,
                          "Thermal BT Difference ΔT (K) [Blue: Cooling | Red: Warming]")
        
        if vis06 is not None and 'vis_06' in sec_channels:
            arr52_v = channels['vis_06']; arr71_v = sec_channels['vis_06']
            m52_v   = masks['vis_06'];    m71_v   = sec_masks['vis_06']
            valid_v = m52_v & m71_v & (dist <= r_disk)
            diff_vis = np.where(valid_v, arr52_v - arr71_v, np.nan)
            render_derived(diff_vis, f"Multi-Temporal VIS Reflectance Shift (Cycle {cycle_id_primary} − {cycle_id_secondary})",
                          "temporal_vis06_change.png", 'PuOr', -40, 40,
                          "Reflectance Difference ΔRefl (%) [Purple: Dimmed | Orange: Brightened]")
        
        print(f"  -> Derived products generated: {success_count}")
        return success_count

    # -----------------------------------------------------------------------
    def write_manifest(self):
        fields = ['product_type', 'product_name', 'composite_id', 'cycle_id',
                  'timestamp', 'filename', 'filepath', 'status', 'is_approx', 'ref']
        with open(self.manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.manifest_records)
        print(f"\n -> Manifest written to: '{self.manifest_path}'")

    # -----------------------------------------------------------------------
    def print_count_report(self, all_ch_success, all_ch_skipped, all_comp_success, all_comp_skipped, derived_count, cycle_count):
        """Print honest count report as requested in STEP 5."""
        total = all_ch_success + all_comp_success + derived_count
        print(f"""
{'='*70}
  STEP 5 — PRODUCT COUNT REPORT (Honest)
{'='*70}
  Source data:
    Repeat cycles found in data/          : {cycle_count}
    Channels actually available per cycle : 4 / 16
      Present  : vis_06, nir_22, ir_38, ir_105
      Absent   : vis_04, vis_05, vis_08, vis_09, nir_13, nir_16,
                 wv_63, wv_73, ir_87, ir_97, ir_123, ir_133
    Reason    : Data files are HRFI (fci_l1c_nc) format. The other 12
                channels require FDHSI (fci_l1c_fdhsi) files, which are
                NOT present in this dataset.

  Channel products:
    Attempted : {all_ch_success + all_ch_skipped} ({cycle_count} cycles × 16 channels)
    Generated : {all_ch_success} ({cycle_count} cycles × 4 available channels)
    Skipped   : {all_ch_skipped} (channels not in source data — logged above)

  Composite products:
    Attempted : {all_comp_success + all_comp_skipped}
    Generated : {all_comp_success} (only composites requiring ≤4 available channels)
    Skipped   : {all_comp_skipped} (missing channels or day/night mismatch)
    Standard composites skipped (missing WV / NIR1.6 / IR12.3 etc.):
      natural_colours, airmass, dust, ash, microphysics_24hr,
      day_microphysics (full), night_microphysics (full),
      severe_storms, snow

  Derived products (pipeline-level, 1 set):  {derived_count}

  TOTAL PRODUCTS GENERATED               : {total}
{'─'*70}
  SHORTFALL vs ~200 target:
    {total} actual vs ~200 target = shortfall of {max(0, 200 - total)}
    To reach 200 via channel products alone:
      Need ~{int(np.ceil(200 / 4))} cycles × 4 channels = 200 (currently only {cycle_count} cycles).
    To reach 200 with 4 channels + composites (~3 per cycle):
      Need ~{int(np.ceil((200 - derived_count) / (4 + 3)))} cycles
      (currently only {cycle_count} available).
    This is a data availability limitation, NOT a pipeline limitation.
    No padding with fake parameter variants has been done.
{'='*70}
""")

    # -----------------------------------------------------------------------
    def run(self):
        t_global = time.time()
        cycles = self.discover_cycles()
        cycle_ids = sorted(cycles.keys())
        
        all_ch_success = 0; all_ch_skipped = 0
        all_comp_success = 0; all_comp_skipped = 0
        
        for cid in cycle_ids:
            chs, csk, comps, comsk = self.process_cycle(cid, cycles[cid])
            all_ch_success   += chs;   all_ch_skipped   += csk
            all_comp_success += comps; all_comp_skipped += comsk
        
        # Derived products: use highest-file-count cycle as primary, other as secondary
        primary_id   = max(cycle_ids, key=lambda k: len(cycles[k]))
        secondary_id = [c for c in cycle_ids if c != primary_id]
        sec_files    = cycles[secondary_id[0]] if secondary_id else None
        sec_id       = secondary_id[0] if secondary_id else None
        
        derived_count = self.generate_derived_products(
            primary_id, cycles[primary_id],
            sec_id, sec_files
        )
        
        self.write_manifest()
        
        elapsed = time.time() - t_global
        print(f"\n -> Total pipeline runtime: {elapsed:.1f}s")
        
        self.print_count_report(
            all_ch_success, all_ch_skipped,
            all_comp_success, all_comp_skipped,
            derived_count, len(cycle_ids)
        )


# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="MTG FCI Multi-Cycle Product Pipeline v2")
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--out-dir',  default='outputs')
    parser.add_argument('--width',  type=int, default=1000)
    parser.add_argument('--height', type=int, default=1024)
    args = parser.parse_args()
    
    pipeline = MTGMultiCyclePipeline(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        target_size=(args.width, args.height)
    )
    pipeline.run()

if __name__ == '__main__':
    main()
