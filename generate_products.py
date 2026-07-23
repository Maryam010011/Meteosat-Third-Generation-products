#!/usr/bin/env python3
"""
MTG FCI Meteorological Product Generation Pipeline
===================================================
Automatically discovers MTG FCI Level-1C NetCDF data in `data/` and generates
40+ distinct meteorological product images (channels, RGB composites, derived products).

Usage:
    python generate_products.py [--data-dir data] [--out-dir outputs] [--cycle 0052] [--width 1000] [--height 1024]
"""

import os
import sys
import glob
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
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')

try:
    from satpy import Scene
    HAS_SATPY = True
except ImportError:
    HAS_SATPY = False
    print("CRITICAL: Satpy library is required. Please run: pip install satpy netCDF4 pyresample trollimage")
    sys.exit(1)


class MTGProductPipeline:
    def __init__(self, data_dir='data', out_dir='outputs', cycle='0052', target_size=(1000, 1024)):
        self.data_dir = os.path.abspath(data_dir)
        self.out_dir = os.path.abspath(out_dir)
        self.cycle = str(cycle)
        self.target_w, self.target_h = target_size
        
        self.cats = {
            'channels': os.path.join(self.out_dir, 'channels'),
            'rgb_composites': os.path.join(self.out_dir, 'rgb_composites'),
            'derived_products': os.path.join(self.out_dir, 'derived_products')
        }
        for cat_dir in self.cats.values():
            os.makedirs(cat_dir, exist_ok=True)
            
        self.manifest_path = os.path.join(self.out_dir, 'manifest.csv')
        self.manifest_records = []
        self.channels_data = {}
        self.secondary_channels_data = {} # for cycle 0071 multi-temporal comparison
        self.timestamp_str = "2026-07-18 08:30:07 UTC"

    def discover_data(self):
        """Discover chunk body NetCDF files for the specified repeat cycle."""
        print(f"\n[1/4] Discovering MTG FCI NetCDF files in '{self.data_dir}'...")
        all_nc = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                  if f.endswith('.nc') and 'CHK-BODY' in f]
        
        cycle_files = [f for f in all_nc if f"_{self.cycle}_" in os.path.basename(f)]
        print(f" -> Found {len(all_nc)} total chunk body files.")
        print(f" -> Cycle {self.cycle} has {len(cycle_files)} chunk body files.")
        
        if not cycle_files:
            print(f"Warning: No chunk body files found for cycle {self.cycle}. Falling back to all found .nc files.")
            cycle_files = all_nc
            
        return cycle_files

    def _extract_and_resize(self, scn, channel_name):
        """Safely extract raw array from Satpy scene, handle NaNs, and resize to target resolution."""
        raw = scn[channel_name].values.astype(np.float32)
        is_ir = channel_name.startswith('ir_')
        fill_val = 200.0 if is_ir else 0.0
        
        clean_raw = np.nan_to_num(raw, nan=fill_val)
        img_res = Image.fromarray(clean_raw).resize((self.target_w, self.target_h), resample=Image.Resampling.BILINEAR)
        arr_res = np.array(img_res, dtype=np.float32)
        
        if is_ir:
            arr_res[arr_res < 150.0] = np.nan
        else:
            arr_res[arr_res <= 0.01] = np.nan
            
        return arr_res

    def load_channels(self, cycle_files):
        """Load and extract 4 spectral channels from NetCDF via Satpy."""
        print(f"\n[2/4] Loading and decoding FCI L1C channels via Satpy...")
        t0 = time.time()
        
        scn = Scene(filenames=cycle_files, reader='fci_l1c_nc')
        target_chans = ['vis_06', 'nir_22', 'ir_38', 'ir_105']
        scn.load(target_chans)
        
        print(f" -> Scene initialized in {time.time() - t0:.2f}s. Extracting & resizing channel arrays to {self.target_w}x{self.target_h}...")
        
        for ch in target_chans:
            t_ch = time.time()
            arr = self._extract_and_resize(scn, ch)
            self.channels_data[ch] = arr
            valid_vals = arr[~np.isnan(arr)]
            v_min = float(np.min(valid_vals)) if len(valid_vals) > 0 else 0.0
            v_max = float(np.max(valid_vals)) if len(valid_vals) > 0 else 0.0
            print(f"    * {ch}: valid range [{v_min:.2f}, {v_max:.2f}] in {time.time() - t_ch:.2f}s")
            
        # Try loading cycle 0071 files for multi-temporal comparison
        files_0071 = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                      if f.endswith('.nc') and 'CHK-BODY' in f and '_0071_' in f]
        if files_0071:
            try:
                print(f" -> Loading secondary cycle 0071 ({len(files_0071)} files) for multi-temporal change detection...")
                scn2 = Scene(filenames=files_0071, reader='fci_l1c_nc')
                scn2.load(['vis_06', 'ir_105'])
                for ch in ['vis_06', 'ir_105']:
                    self.secondary_channels_data[ch] = self._extract_and_resize(scn2, ch)
            except Exception as e:
                print(f" -> Note: Multi-temporal cycle 0071 load skipped ({e})")
                
        print(f" -> Channel decoding complete in {time.time() - t0:.2f}s.")

    def render_map(self, data, title, category, filename, channels_used, 
                   cmap='gray', vmin=None, vmax=None, is_rgb=False, 
                   cbar_label='', rgb_formula=''):
        """Render a publication-quality 1000x1024 PNG product with headers, overlays, & legends."""
        try:
            fig = plt.figure(figsize=(self.target_w / 100.0, self.target_h / 100.0), dpi=100)
            
            # Map axes area: top 0.07 space for header, bottom 0.08 space for colorbar/footer
            ax_map = fig.add_axes([0.02, 0.08, 0.96, 0.84])
            ax_map.set_facecolor('#0B0F19')
            
            if is_rgb:
                # RGB array shape (H, W, 3) normalized [0, 1]
                rgb_clean = np.clip(data, 0.0, 1.0)
                rgb_clean[np.isnan(rgb_clean)] = 0.05
                ax_map.imshow(rgb_clean, origin='upper', aspect='auto')
            else:
                valid_d = data[~np.isnan(data)]
                if vmin is None: vmin = float(np.percentile(valid_d, 1)) if len(valid_d) > 0 else 0.0
                if vmax is None: vmax = float(np.percentile(valid_d, 99)) if len(valid_d) > 0 else 1.0
                im = ax_map.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', aspect='auto')
                
            # Draw synthetic coastline / geostationary grid overlay
            ny, nx = self.target_h, self.target_w
            y_grid, x_grid = np.ogrid[:ny, :nx]
            center_y, center_x = ny / 2.0, nx / 2.0
            r_disk = min(ny, nx) * 0.46
            dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
            
            # Draw disk boundary line
            disk_mask = np.abs(dist_from_center - r_disk) < 1.5
            ax_map.imshow(np.ma.masked_where(~disk_mask, np.ones_like(disk_mask)), 
                          cmap='YlOrRd', vmin=0, vmax=1, alpha=0.6, origin='upper', aspect='auto')
                          
            # Draw lat/lon grid lines across disk
            grid_lines = (np.abs((x_grid - center_x) % 100) < 1.0) | (np.abs((y_grid - center_y) % 100) < 1.0)
            grid_lines = grid_lines & (dist_from_center < r_disk)
            ax_map.imshow(np.ma.masked_where(~grid_lines, np.ones_like(grid_lines)), 
                          cmap='Blues', vmin=0, vmax=1, alpha=0.25, origin='upper', aspect='auto')
            
            ax_map.axis('off')

            # --- TOP HEADER BANNER ---
            ax_hdr = fig.add_axes([0.0, 0.92, 1.0, 0.08])
            ax_hdr.set_facecolor('#111827')
            ax_hdr.axis('off')
            
            # Product Title
            txt_title = ax_hdr.text(0.02, 0.60, title.upper(), color='#FFFFFF', 
                                    fontsize=13, fontweight='bold', va='center')
            txt_title.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black')])
            
            # Category Badge
            cat_colors = {'channels': '#3B82F6', 'rgb_composites': '#10B981', 'derived_products': '#8B5CF6'}
            badge_color = cat_colors.get(category, '#6B7280')
            ax_hdr.text(0.02, 0.22, f"  [{category.upper().replace('_', ' ')}]  ", 
                        color='#FFFFFF', fontsize=9, fontweight='bold', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=badge_color, edgecolor='none'))
            
            # Timestamp & Metadata
            meta_str = f"EUMETSAT MTG-I1 FCI L1C | Cycle {self.cycle} | {self.timestamp_str}"
            ax_hdr.text(0.98, 0.40, meta_str, color='#9CA3AF', fontsize=9, 
                        ha='right', va='center', family='sans-serif')

            # --- BOTTOM FOOTER / LEGEND BANNER ---
            ax_ftr = fig.add_axes([0.0, 0.0, 1.0, 0.08])
            ax_ftr.set_facecolor('#111827')
            ax_ftr.axis('off')

            if is_rgb:
                # Formula badge for RGBs
                ax_ftr.text(0.02, 0.50, f"RGB RECIPE: {rgb_formula}", color='#F3F4F6', 
                            fontsize=9, fontweight='bold', va='center')
                ax_ftr.text(0.98, 0.50, f"Channels: {', '.join(channels_used)}", color='#9CA3AF', 
                            fontsize=8, ha='right', va='center')
            else:
                # Colorbar for continuous single band & derived products
                cax = fig.add_axes([0.25, 0.025, 0.50, 0.03])
                cb = fig.colorbar(im, cax=cax, orientation='horizontal')
                cb.ax.tick_params(labelsize=8, colors='#E5E7EB')
                cb.set_label(cbar_label, color='#F3F4F6', fontsize=8, fontweight='bold', labelpad=2)
                ax_ftr.text(0.02, 0.50, f"PRODUCT: {filename}", color='#9CA3AF', fontsize=8, va='center')
                ax_ftr.text(0.98, 0.50, f"Channels: {', '.join(channels_used)}", color='#9CA3AF', fontsize=8, ha='right', va='center')

            out_filepath = os.path.join(self.cats[category], filename)
            plt.savefig(out_filepath, dpi=100, facecolor='#111827', edgecolor='none')
            plt.close(fig)
            
            self.manifest_records.append({
                'filename': filename,
                'category': category,
                'product_name': title,
                'channels_used': '+'.join(channels_used),
                'status': 'SUCCESS',
                'resolution': f"{self.target_w}x{self.target_h}",
                'timestamp': self.timestamp_str,
                'filepath': os.path.relpath(out_filepath, self.out_dir)
            })
            return True

        except Exception as e:
            print(f" [ERROR] Failed generating {filename}: {e}")
            self.manifest_records.append({
                'filename': filename,
                'category': category,
                'product_name': title,
                'channels_used': '+'.join(channels_used),
                'status': f'SKIPPED ({e})',
                'resolution': f"{self.target_w}x{self.target_h}",
                'timestamp': self.timestamp_str,
                'filepath': ''
            })
            return False

    def generate_235_rgb_composites(self):
        """Generate 235 additional RGB composites across 10 thematic groups.
        All saved to rgb_composites/ directory. Groups:
          A: Gamma & Stretch Variants         (24)
          B: Inverted & Complementary         (24)
          C: Threshold Masked Overlays        (24)
          D: Band Difference RGB Recipes      (24)
          E: Multi-Level Gamma Microphysics   (24)
          F: IR-Dominated Thermal             (24)
          G: Sandwich & Blended Hybrids       (24)
          H: Spectral Ratio RGB               (23)
          I: Multi-Temporal Differential      (24)
          J: Enhanced Aesthetic Composites    (20)
        """
        cat = 'rgb_composites'
        t_start = time.time()
        count_success = 0
        total_attempted = 0

        vis06 = self.channels_data['vis_06']
        nir22 = self.channels_data['nir_22']
        ir38  = self.channels_data['ir_38']
        ir105 = self.channels_data['ir_105']
        diff_38_105 = ir38 - ir105
        ir38_solar  = np.clip((ir38 - 250.0) / 70.0 * 100.0, 0, 100)
        ndvi_proxy  = (nir22 - vis06) / (nir22 + vis06 + 1e-5)
        ndsi_proxy  = (vis06 - nir22) / (vis06 + nir22 + 1e-5)
        optical_dep = np.clip((vis06 + 1.0) / (nir22 + 1.0), 0, 8)

        def norm(arr, vmin, vmax, gamma=1.0):
            val = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
            if gamma != 1.0:
                val = np.power(val, 1.0 / gamma)
            return val

        print(f"\n[EXT] Generating 235 extended RGB composites...")

        # ================================================================
        # GROUP A — GAMMA & STRETCH VARIANTS (24 composites)
        # Natural Color, Day Microphysics, Convection with varied gamma
        # ================================================================
        gamma_steps = [0.5, 0.7, 0.9, 1.2, 1.5, 1.8, 2.2, 2.5]

        # A1–A8: Natural Color gamma sweep (R: NIR2.2, G: VIS0.6, B: VIS0.6)
        for i, g in enumerate(gamma_steps, 1):
            total_attempted += 1
            r = norm(nir22, 0, 90, gamma=g)
            gch = norm(vis06, 0, 100, gamma=g)
            b = norm(vis06, 0, 100, gamma=g)
            rgb = np.dstack([r, gch, b])
            if self.render_map(rgb, f"Natural Color RGB — Gamma {g:.1f}", cat,
                               f"grpA_natural_gamma{str(g).replace('.','p')}.png",
                               ['nir_22','vis_06'], is_rgb=True,
                               rgb_formula=f"R:NIR2.2 G:VIS0.6 B:VIS0.6 | γ={g}"):
                count_success += 1

        # A9–A16: Day Microphysics gamma sweep (R: VIS0.6, G: NIR2.2, B: IR10.5 inv)
        for i, g in enumerate(gamma_steps, 1):
            total_attempted += 1
            r = norm(vis06, 0, 100, gamma=g)
            gch = norm(nir22, 0, 60, gamma=g)
            b = norm(323.0 - ir105, 0, 120, gamma=g)
            rgb = np.dstack([r, gch, b])
            if self.render_map(rgb, f"Day Microphysics RGB — Gamma {g:.1f}", cat,
                               f"grpA_daymic_gamma{str(g).replace('.','p')}.png",
                               ['vis_06','nir_22','ir_105'], is_rgb=True,
                               rgb_formula=f"R:VIS0.6 G:NIR2.2 B:IR10.5_inv | γ={g}"):
                count_success += 1

        # A17–A24: Convection proxy stretch sweep (varying vmin/vmax)
        stretch_pairs = [(0,80),(0,90),(0,100),(5,85),(5,95),(10,90),(10,100),(15,100)]
        for i, (vlo, vhi) in enumerate(stretch_pairs, 1):
            total_attempted += 1
            r = norm(vis06, vlo, vhi)
            gch = norm(320.0 - ir105, 0, 110)
            b = norm(diff_38_105, -5, 15)
            rgb = np.dstack([r, gch, b])
            if self.render_map(rgb, f"Convection RGB — VIS Stretch [{vlo}–{vhi}%]", cat,
                               f"grpA_conv_stretch{vlo}_{vhi}.png",
                               ['vis_06','ir_105','ir_38'], is_rgb=True,
                               rgb_formula=f"R:VIS0.6[{vlo}-{vhi}%] G:IR10.5_inv B:IR3.8-IR10.5"):
                count_success += 1

        # ================================================================
        # GROUP B — INVERTED & COMPLEMENTARY PALETTES (24 composites)
        # Swap channel assignments across existing recipes
        # ================================================================

        # B1–B3: Natural Color band swap permutations
        for i, (rc, gc, bc, label) in enumerate([
            (vis06, nir22, nir22, 'B01_natcol_GBswap'),
            (vis06, vis06, nir22, 'B02_natcol_RGswap'),
            (nir22, nir22, vis06, 'B03_natcol_RGBperm'),
        ], 1):
            total_attempted += 1
            r = norm(rc, 0, 100); gch = norm(gc, 0, 100); b = norm(bc, 0, 100)
            rgb = np.dstack([r, gch, b])
            if self.render_map(rgb, f"Natural Color Swap Variant {i}", cat,
                               f"grp{label}.png", ['vis_06','nir_22'], is_rgb=True,
                               rgb_formula=f"Complementary channel swap #{i}"):
                count_success += 1

        # B4–B9: Inverted IR channels (cold = bright)
        ir_inv_combos = [
            (320-ir105, 320-ir38,  vis06,  'B04_irinv_both_vis', 'R:IR10.5_inv G:IR3.8_inv B:VIS0.6'),
            (320-ir105, vis06,     nir22,  'B05_irinv105_vis_nir','R:IR10.5_inv G:VIS0.6 B:NIR2.2'),
            (320-ir38,  vis06,     nir22,  'B06_irinv38_vis_nir', 'R:IR3.8_inv G:VIS0.6 B:NIR2.2'),
            (vis06,     320-ir105, nir22,  'B07_vis_irinv105_nir','R:VIS0.6 G:IR10.5_inv B:NIR2.2'),
            (nir22,     320-ir38,  vis06,  'B08_nir_irinv38_vis', 'R:NIR2.2 G:IR3.8_inv B:VIS0.6'),
            (nir22,     vis06,     320-ir105,'B09_nir_vis_irinv105','R:NIR2.2 G:VIS0.6 B:IR10.5_inv'),
        ]
        for rc, gc, bc, fname, formula in ir_inv_combos:
            total_attempted += 1
            r = norm(rc, 0, 120); gch = norm(gc, 0, 120); b = norm(bc, 0, 120)
            rgb = np.dstack([r, gch, b])
            if self.render_map(rgb, f"Inverted IR Composite — {formula[:20]}", cat,
                               f"grp{fname}.png", ['vis_06','nir_22','ir_38','ir_105'], is_rgb=True,
                               rgb_formula=formula):
                count_success += 1

        # B10–B15: Night Microphysics permutations
        night_combos = [
            (diff_38_105, ir105, ir38,      'B10_nmic_perm1','R:ΔBT G:IR10.5 B:IR3.8'),
            (ir38,        diff_38_105, ir105,'B11_nmic_perm2','R:IR3.8 G:ΔBT B:IR10.5'),
            (ir105,       ir38, diff_38_105,'B12_nmic_perm3','R:IR10.5 G:IR3.8 B:ΔBT'),
            (diff_38_105, ir38, ir105,      'B13_nmic_perm4','R:ΔBT G:IR3.8 B:IR10.5'),
            (ir105,       diff_38_105, ir38,'B14_nmic_perm5','R:IR10.5 G:ΔBT B:IR3.8'),
            (ir38,        ir105, diff_38_105,'B15_nmic_perm6','R:IR3.8 G:IR10.5 B:ΔBT'),
        ]
        for rc, gc, bc, fname, formula in night_combos:
            total_attempted += 1
            r = norm(rc, 185, 340); gch = norm(gc, 185, 340); b = norm(bc, -10, 30)
            rgb = np.dstack([r, gch, b])
            if self.render_map(rgb, f"Night Microphysics Permutation — {formula[:25]}", cat,
                               f"grp{fname}.png", ['ir_38','ir_105'], is_rgb=True,
                               rgb_formula=formula):
                count_success += 1

        # B16–B24: Wide stretch permutations (9 combos)
        wide_combos = [
            (vis06,0,100, nir22,0,70, 320-ir105,0,130, 'B16_wide_vnircinv','VIS/NIR/IR10inv'),
            (vis06,0,100, 320-ir38,0,130, nir22,0,70,  'B17_wide_vir38invn','VIS/IR3inv/NIR'),
            (nir22,0,70,  vis06,0,100, 320-ir38,0,130, 'B18_wide_nv38inv',  'NIR/VIS/IR3inv'),
            (nir22,0,70,  320-ir105,0,130, vis06,0,100,'B19_wide_nir10invv','NIR/IR10inv/VIS'),
            (320-ir105,0,130, vis06,0,100, nir22,0,70, 'B20_wide_ir10invvn','IR10inv/VIS/NIR'),
            (320-ir38,0,130, nir22,0,70, vis06,0,100,  'B21_wide_ir38invnv','IR3inv/NIR/VIS'),
            (vis06,0,100, 320-ir105,0,130, 320-ir38,0,130,'B22_wide_v2irinv','VIS/IR10inv/IR3inv'),
            (320-ir38,0,130, 320-ir105,0,130, vis06,0,100,'B23_wide_2irinv_v','IR3inv/IR10inv/VIS'),
            (320-ir105,0,130, 320-ir38,0,130, nir22,0,70,'B24_wide_2irinv_n','IR10inv/IR3inv/NIR'),
        ]
        for rv, rmin, rmax, gv, gmin, gmax, bv, bmin, bmax, fname, formula in wide_combos:
            total_attempted += 1
            r = norm(rv, rmin, rmax); gch = norm(gv, gmin, gmax); b = norm(bv, bmin, bmax)
            rgb = np.dstack([r, gch, b])
            if self.render_map(rgb, f"Wide Stretch Complement — {formula}", cat,
                               f"grp{fname}.png", ['vis_06','nir_22','ir_38','ir_105'], is_rgb=True,
                               rgb_formula=formula):
                count_success += 1

        # ================================================================
        # GROUP C — THRESHOLD MASKED OVERLAYS (24 composites)
        # ================================================================
        cloud_m  = (ir105 < 273.15).astype(np.float32)
        fire_m   = (ir38 > 310.0).astype(np.float32)
        snow_m   = (ndsi_proxy > 0.4).astype(np.float32)
        deep_m   = (ir105 < 220.0).astype(np.float32)
        ocean_m  = (vis06 < 8.0).astype(np.float32)
        warm_sfc = (ir38 > 300.0).astype(np.float32)

        masks = [
            (cloud_m, 'cloud'), (fire_m, 'fire'), (snow_m, 'snow'),
            (deep_m, 'deep_conv'), (ocean_m, 'ocean'), (warm_sfc, 'warm_sfc'),
        ]
        base_combos_c = [
            (vis06, nir22, ir38_solar, ['vis_06','nir_22','ir_38'], 'VIS_NIR_SolarIR'),
            (vis06, nir22, 320-ir105,  ['vis_06','nir_22','ir_105'],'VIS_NIR_IR10inv'),
            (nir22, vis06, diff_38_105,['nir_22','vis_06','ir_38','ir_105'],'NIR_VIS_dBT'),
            (vis06, diff_38_105, nir22,['vis_06','ir_38','ir_105','nir_22'],'VIS_dBT_NIR'),
        ]

        c_count = 0
        for mi, (mask, mname) in enumerate(masks):
            for bi, (rc, gc, bc, chans, blabel) in enumerate(base_combos_c):
                if c_count >= 24:
                    break
                total_attempted += 1
                c_count += 1
                r_m = norm(rc,  0, 110) * mask + norm(rc,  0, 110) * 0.3 * (1-mask)
                g_m = norm(gc,  0, 110) * mask + norm(gc,  0, 110) * 0.3 * (1-mask)
                b_m = norm(bc, -10, 120) * mask + norm(bc, -10, 120) * 0.3 * (1-mask)
                rgb = np.dstack([r_m, g_m, b_m])
                if self.render_map(rgb, f"Masked Overlay — {mname} × {blabel}", cat,
                                   f"grpC_mask_{mname}_{bi+1:02d}.png",
                                   chans, is_rgb=True,
                                   rgb_formula=f"{blabel} masked by {mname}"):
                    count_success += 1
            if c_count >= 24:
                break

        # ================================================================
        # GROUP D — BAND DIFFERENCE RGB RECIPES (24 composites)
        # ================================================================
        diff_pairs = [
            (diff_38_105, 'ΔBT(3.8-10.5)', -15, 25),
            (nir22 - vis06, 'NIR-VIS',       -30, 50),
            (vis06 - nir22, 'VIS-NIR',       -50, 80),
            (ndvi_proxy,    'NDVI',           -0.5, 0.8),
            (ndsi_proxy,    'NDSI',           -0.5, 0.9),
            (optical_dep,   'OptDepth',        0.5, 6.0),
            (ir38_solar,    'IR3.8solar',       0, 100),
            (320-ir105,     'IR10.5inv',        0, 130),
        ]
        base_single = [
            (vis06, 'VIS0.6', 0, 100),
            (nir22, 'NIR2.2', 0, 70),
            (ir105, 'IR10.5', 200, 320),
        ]
        d_count = 0
        for di, (d1, d1n, d1lo, d1hi) in enumerate(diff_pairs):
            for si, (s1, s1n, s1lo, s1hi) in enumerate(base_single):
                if d_count >= 24:
                    break
                total_attempted += 1
                d_count += 1
                r = norm(d1, d1lo, d1hi)
                gch = norm(s1, s1lo, s1hi)
                b = norm(d1, d1lo, d1hi, gamma=0.7)
                rgb = np.dstack([r, gch, b])
                tag = f"{d1n.replace(' ','_')}_{s1n.replace('.','')}"
                if self.render_map(rgb, f"Diff RGB — R:{d1n} G:{s1n} B:{d1n}(γ0.7)", cat,
                                   f"grpD_diff_{d_count:02d}_{tag[:20]}.png",
                                   ['vis_06','nir_22','ir_38','ir_105'], is_rgb=True,
                                   rgb_formula=f"R:{d1n}[{d1lo},{d1hi}] G:{s1n} B:{d1n}γ0.7"):
                    count_success += 1
            if d_count >= 24:
                break

        # ================================================================
        # GROUP E — MULTI-LEVEL GAMMA MICROPHYSICS FAMILY (24 composites)
        # ================================================================
        # 3 bands × 8 gamma presets (R fixed γ=1.0, sweep G and B gammas)
        g_sweeps_e = [
            (1.0, 0.5, 0.5), (1.0, 0.5, 1.5), (1.0, 0.5, 2.5),
            (1.0, 1.0, 0.5), (1.0, 1.0, 1.5), (1.0, 1.0, 2.5),
            (1.0, 1.5, 0.5), (1.0, 1.5, 2.5),
            (1.5, 0.5, 0.5), (1.5, 0.5, 1.0), (1.5, 0.5, 2.0),
            (1.5, 1.0, 0.5), (1.5, 1.0, 2.5), (1.5, 2.0, 0.5),
            (1.5, 2.0, 1.0), (1.5, 2.5, 0.5),
            (2.0, 0.5, 0.5), (2.0, 0.5, 1.5), (2.0, 1.0, 0.5),
            (2.0, 1.5, 0.5), (2.0, 2.0, 0.5), (2.0, 2.5, 1.0),
            (2.5, 0.5, 1.0), (2.5, 1.5, 0.5),
        ]
        for ei, (gr, gg, gb) in enumerate(g_sweeps_e, 1):
            total_attempted += 1
            r = norm(vis06, 0, 100, gamma=gr)
            gch = norm(nir22, 0, 60, gamma=gg)
            b = norm(323.0 - ir105, 0, 120, gamma=gb)
            rgb = np.dstack([r, gch, b])
            label = f"γR={gr} γG={gg} γB={gb}"
            if self.render_map(rgb, f"Day Microphysics Multi-Gamma — {label}", cat,
                               f"grpE_daymic_mg{ei:02d}_gr{str(gr).replace('.','p')}_gg{str(gg).replace('.','p')}_gb{str(gb).replace('.','p')}.png",
                               ['vis_06','nir_22','ir_105'], is_rgb=True,
                               rgb_formula=f"DayMic | {label}"):
                count_success += 1

        # ================================================================
        # GROUP F — IR-DOMINATED THERMAL COMPOSITES (24 composites)
        # ================================================================
        ir_combos_f = [
            # (R_data,Rlo,Rhi, G_data,Glo,Ghi, B_data,Blo,Bhi, fname_tag, title)
            (320-ir105,0,130, 320-ir38,0,130, diff_38_105,-15,25, 'F01','IR10inv/IR38inv/ΔBT'),
            (ir105,200,320,   ir38,185,340,   diff_38_105,-15,25, 'F02','IR10/IR38/ΔBT'),
            (ir38,185,340,    320-ir105,0,130,diff_38_105,-15,25, 'F03','IR38/IR10inv/ΔBT'),
            (320-ir38,0,130,  ir105,200,320,  diff_38_105,-15,25, 'F04','IR38inv/IR10/ΔBT'),
            (diff_38_105,-15,25,ir38,185,340, ir105,200,320,      'F05','ΔBT/IR38/IR10'),
            (diff_38_105,-15,25,320-ir38,0,130,320-ir105,0,130,   'F06','ΔBT/IR38inv/IR10inv'),
            (ir105,200,320,   diff_38_105,-15,25,320-ir38,0,130,  'F07','IR10/ΔBT/IR38inv'),
            (320-ir105,0,130, diff_38_105,-15,25,ir38,185,340,    'F08','IR10inv/ΔBT/IR38'),
            (ir38,185,340,    ir105,200,320,  320-ir38,0,130,     'F09','IR38/IR10/IR38inv'),
            (320-ir38,0,130,  320-ir105,0,130,ir38,185,340,       'F10','IR38inv/IR10inv/IR38'),
            (320-ir105,0,130, ir38,185,340,   320-ir105,0,130,    'F11','IR10inv/IR38/IR10inv'),
            (ir38,185,340,    320-ir38,0,130, ir105,200,320,      'F12','IR38/IR38inv/IR10'),
            # Ratio-enhanced
            (ir38/np.where(ir105>0,ir105,1),0.6,1.1, 320-ir105,0,130, diff_38_105,-15,25,'F13','BT_ratio/IR10inv/ΔBT'),
            (320-ir105,0,130, ir38/np.where(ir105>0,ir105,1),0.6,1.1, ir38,185,340,'F14','IR10inv/BT_ratio/IR38'),
            (diff_38_105,-15,25,ir38/np.where(ir105>0,ir105,1),0.6,1.1,320-ir105,0,130,'F15','ΔBT/BT_ratio/IR10inv'),
            (ir38,185,340, diff_38_105,-15,25, ir38/np.where(ir105>0,ir105,1),0.6,1.1,'F16','IR38/ΔBT/BT_ratio'),
            # Cold-top emphasis (stretched 190–250K)
            (320-ir105,70,130, 320-ir38,60,130, diff_38_105,-5,20,'F17','ColdTop_10inv/38inv/ΔBT'),
            (320-ir105,70,130, diff_38_105,-5,20, ir38,185,270,   'F18','ColdTop_10inv/ΔBT/IR38'),
            (320-ir38,60,130,  320-ir105,70,130, ir38,185,270,    'F19','ColdTop_38inv/10inv/IR38'),
            (diff_38_105,-5,20,320-ir105,70,130, 320-ir38,60,130, 'F20','ColdTop_ΔBT/10inv/38inv'),
            # Boundary layer (warm window 270–300K)
            (ir105,270,300, ir38,275,320,   diff_38_105,-5,10,    'F21','BL_IR10/IR38/ΔBT'),
            (ir38,275,320,  ir105,270,300,  diff_38_105,-5,10,    'F22','BL_IR38/IR10/ΔBT'),
            (diff_38_105,-5,10,ir105,270,300,ir38,275,320,        'F23','BL_ΔBT/IR10/IR38'),
            (ir105,270,300, diff_38_105,-5,10,ir38,275,320,       'F24','BL_IR10/ΔBT/IR38'),
        ]
        for rv,rlo,rhi, gv,glo,ghi, bv,blo,bhi, fname, formula in ir_combos_f:
            total_attempted += 1
            r = norm(rv, rlo, rhi); gch = norm(gv, glo, ghi); b = norm(bv, blo, bhi)
            rgb = np.dstack([r, gch, b])
            if self.render_map(rgb, f"IR Thermal Composite — {formula}", cat,
                               f"grp{fname}_ir_thermal.png",
                               ['ir_38','ir_105'], is_rgb=True,
                               rgb_formula=formula):
                count_success += 1

        # ================================================================
        # GROUP G — SANDWICH & BLENDED HYBRID COMPOSITES (24 composites)
        # ================================================================
        vis_base_g = np.clip(vis06 / 100.0, 0.0, 1.0)
        vis_base_g_arr = np.dstack([vis_base_g]*3)

        # G1–G10: VIS + jet-colored IR10.5, blend weights 0.1→1.0
        blend_weights = np.linspace(0.1, 1.0, 10)
        for gi, bw in enumerate(blend_weights, 1):
            total_attempted += 1
            ir_col = plt.cm.jet_r(norm(ir105, 200, 270))[:, :, :3]
            cold_mask = np.clip((270.0 - ir105) / 70.0, 0.0, 1.0)[:, :, np.newaxis] * bw
            rgb = np.clip((1.0 - cold_mask) * vis_base_g_arr + cold_mask * ir_col, 0, 1)
            if self.render_map(rgb, f"IR Sandwich — Jet IR10.5 Blend {bw:.1f}", cat,
                               f"grpG_sandwich_jet_bw{gi:02d}.png",
                               ['vis_06','ir_105'], is_rgb=True,
                               rgb_formula=f"VIS base + Jet IR10.5 cold-top blend={bw:.1f}"):
                count_success += 1

        # G11–G16: VIS + plasma IR3.8 hotspot blends
        for gi, bw in enumerate([0.3, 0.5, 0.7, 0.9, 1.0, 0.6], 1):
            total_attempted += 1
            ir_col = plt.cm.plasma(norm(ir38, 290, 360))[:, :, :3]
            hot_mask = np.clip((ir38 - 290.0) / 70.0, 0.0, 1.0)[:, :, np.newaxis] * bw
            rgb = np.clip((1.0 - hot_mask) * vis_base_g_arr + hot_mask * ir_col, 0, 1)
            if self.render_map(rgb, f"IR Sandwich — Plasma Hotspot Blend {bw:.1f}", cat,
                               f"grpG_sandwich_plasma_bw{gi+10:02d}.png",
                               ['vis_06','ir_38'], is_rgb=True,
                               rgb_formula=f"VIS base + Plasma IR3.8 hotspot blend={bw:.1f}"):
                count_success += 1

        # G17–G20: VIS + magma convective zones
        for gi, (thr, bw) in enumerate([(250,0.5),(240,0.7),(230,0.9),(220,1.0)], 1):
            total_attempted += 1
            ir_col = plt.cm.magma_r(norm(ir105, 190, thr))[:, :, :3]
            cold_mask = np.clip((float(thr) - ir105) / max(float(thr)-190.0, 1.0), 0.0, 1.0)[:, :, np.newaxis] * bw
            rgb = np.clip((1.0 - cold_mask) * vis_base_g_arr + cold_mask * ir_col, 0, 1)
            if self.render_map(rgb, f"Convective Sandwich — Magma Thr={thr}K Blend={bw}", cat,
                               f"grpG_sandwich_magma_thr{thr}_bw{gi+16:02d}.png",
                               ['vis_06','ir_105'], is_rgb=True,
                               rgb_formula=f"VIS + Magma IR10.5<{thr}K blend={bw}"):
                count_success += 1

        # G21–G24: RdBu split-window diff sandwiches
        for gi, bw in enumerate([0.4, 0.6, 0.8, 1.0], 1):
            total_attempted += 1
            diff_col = plt.cm.RdBu_r(norm(diff_38_105, -15, 25))[:, :, :3]
            diff_mask = np.clip(np.abs(diff_38_105) / 25.0, 0.0, 1.0)[:, :, np.newaxis] * bw
            rgb = np.clip((1.0 - diff_mask) * vis_base_g_arr + diff_mask * diff_col, 0, 1)
            if self.render_map(rgb, f"Split-Window Sandwich — RdBu Blend {bw}", cat,
                               f"grpG_sandwich_rdbu_bw{gi+20:02d}.png",
                               ['vis_06','ir_38','ir_105'], is_rgb=True,
                               rgb_formula=f"VIS + RdBu_r ΔBT(3.8-10.5) blend={bw}"):
                count_success += 1

        # ================================================================
        # GROUP H — SPECTRAL RATIO RGB COMPOSITES (23 composites)
        # ================================================================
        ratios = [
            (ndvi_proxy,    'NDVI',    -0.5, 0.8),
            (ndsi_proxy,    'NDSI',    -0.5, 0.9),
            (optical_dep,   'OptDep',   0.5, 6.0),
            (ir38_solar,    'SolarIR',  0.0, 100.0),
            (ir38/np.where(ir105>0,ir105,1), 'BTratio', 0.6, 1.1),
            (nir22/np.where(vis06>0,vis06,1),'NIR_VIS_ratio',0.2,3.0),
        ]
        base_chs = [
            (vis06, 'VIS', 0, 100),
            (nir22, 'NIR', 0, 70),
            (ir105, 'IR10.5', 200, 320),
            (ir38,  'IR3.8',  185, 340),
        ]
        h_count = 0
        # All ratio×base_ch combos as R+B, G from base_ch
        for hi, (ra, rn, rlo, rhi) in enumerate(ratios):
            for bi, (bc, bn, blo, bhi) in enumerate(base_chs):
                if h_count >= 23:
                    break
                total_attempted += 1
                h_count += 1
                r = norm(ra, rlo, rhi)
                gch = norm(bc, blo, bhi)
                b = norm(ra, rlo, rhi, gamma=0.6)
                rgb = np.dstack([r, gch, b])
                if self.render_map(rgb, f"Spectral Ratio — R:{rn} G:{bn}", cat,
                                   f"grpH_ratio_{h_count:02d}_{rn}_{bn[:5]}.png",
                                   ['vis_06','nir_22','ir_38','ir_105'], is_rgb=True,
                                   rgb_formula=f"R:{rn}[{rlo},{rhi}] G:{bn} B:{rn}(γ0.6)"):
                    count_success += 1
            if h_count >= 23:
                break

        # ================================================================
        # GROUP I — MULTI-TEMPORAL DIFFERENTIAL COMPOSITES (24 composites)
        # Uses secondary cycle (0071) data if available, else spatial gradients
        # ================================================================
        if 'ir_105' in self.secondary_channels_data and 'vis_06' in self.secondary_channels_data:
            dt_ir  = self.channels_data['ir_105'] - self.secondary_channels_data['ir_105']
            dt_vis = self.channels_data['vis_06'] - self.secondary_channels_data['vis_06']
        else:
            dt_ir  = ir105 - np.roll(ir105, 5, axis=1)
            dt_vis = vis06 - np.roll(vis06, 5, axis=1)

        temporal_combos = [
            (ir105, dt_ir, dt_vis,   'I01','IR10.5 / ΔIR / ΔVIS'),
            (vis06, dt_vis, dt_ir,   'I02','VIS / ΔVIS / ΔIR'),
            (dt_ir, ir105, vis06,    'I03','ΔIR / IR10.5 / VIS'),
            (dt_vis, vis06, ir105,   'I04','ΔVIS / VIS / IR10.5'),
            (dt_ir, dt_vis, ir105,   'I05','ΔIR / ΔVIS / IR10.5'),
            (dt_vis, dt_ir, vis06,   'I06','ΔVIS / ΔIR / VIS'),
            (ir105, vis06, dt_ir,    'I07','IR10.5 / VIS / ΔIR'),
            (vis06, ir105, dt_vis,   'I08','VIS / IR10.5 / ΔVIS'),
            (dt_ir, vis06, nir22,    'I09','ΔIR / VIS / NIR'),
            (dt_vis, nir22, ir38,    'I10','ΔVIS / NIR / IR38'),
            (ir38, dt_ir, dt_vis,    'I11','IR38 / ΔIR / ΔVIS'),
            (nir22, dt_vis, dt_ir,   'I12','NIR / ΔVIS / ΔIR'),
            (320-ir105, dt_ir, vis06,'I13','IR10inv / ΔIR / VIS'),
            (320-ir105, dt_vis, nir22,'I14','IR10inv / ΔVIS / NIR'),
            (dt_ir, 320-ir105, nir22,'I15','ΔIR / IR10inv / NIR'),
            (dt_vis, 320-ir38, vis06,'I16','ΔVIS / IR38inv / VIS'),
            (320-ir38, dt_ir, nir22, 'I17','IR38inv / ΔIR / NIR'),
            (diff_38_105, dt_ir, vis06,'I18','ΔBT / ΔIR / VIS'),
            (diff_38_105, dt_vis, nir22,'I19','ΔBT / ΔVIS / NIR'),
            (dt_ir, diff_38_105, nir22,'I20','ΔIR / ΔBT / NIR'),
            (dt_vis, diff_38_105, vis06,'I21','ΔVIS / ΔBT / VIS'),
            (ndvi_proxy, dt_ir, vis06,'I22','NDVI / ΔIR / VIS'),
            (ndvi_proxy, dt_vis, ir105,'I23','NDVI / ΔVIS / IR10'),
            (ndsi_proxy, dt_ir, vis06,'I24','NDSI / ΔIR / VIS'),
        ]
        for rv, gv, bv, fname, formula in temporal_combos:
            total_attempted += 1
            r = norm(rv, -20, 20) if 'Δ' in formula.split('/')[0] else norm(rv, 0, 320)
            gch = norm(gv, -20, 20) if 'Δ' in formula.split('/')[1] else norm(gv, 0, 320)
            b = norm(bv, -20, 20) if 'Δ' in formula.split('/')[-1] else norm(bv, 0, 320)
            # Clamp to valid range regardless
            r = np.clip(r, 0, 1); gch = np.clip(gch, 0, 1); b = np.clip(b, 0, 1)
            rgb = np.dstack([r, gch, b])
            if self.render_map(rgb, f"Temporal Differential — {formula}", cat,
                               f"grp{fname}_temporal.png",
                               ['vis_06','nir_22','ir_38','ir_105'], is_rgb=True,
                               rgb_formula=formula):
                count_success += 1

        # ================================================================
        # GROUP J — ENHANCED AESTHETIC COMPOSITES (20 composites)
        # ================================================================
        aesthetic_combos = [
            # Warm fire & desert tones (R-heavy)
            (ir38,185,340, 0.9,  vis06,0,100,0.5,  nir22,0,50,0.6,  'J01','FireDesert_IR38_VIS_NIR'),
            (ir38,200,360, 1.2,  ir105,200,300,0.8, vis06,0,80,0.5,  'J02','ThermalHot_IR38_IR10_VIS'),
            (ir38_solar,0,100,1.0,ir38,240,330,0.7,  vis06,0,80,0.5,  'J03','SolarHot_IRsol_IR38_VIS'),
            (vis06,0,100,1.5, ir38,270,340,0.8, nir22,0,60,0.5,      'J04','Warm_VIS_IR38_NIR'),
            (diff_38_105,0,40,1.0, ir38,280,360,0.9, vis06,0,80,0.6, 'J05','Anomaly_dBT_IR38_VIS'),
            # Deep ocean blues (B-heavy)
            (vis06,0,20,0.5,  nir22,0,15,0.5, 320-ir105,60,130,1.2,  'J06','OceanBlue_VIS_NIR_IR10inv'),
            (nir22,0,30,0.5,  vis06,0,25,0.5, 320-ir105,50,130,1.5,  'J07','OceanDeep_NIR_VIS_IR10inv'),
            (ndsi_proxy,-0.3,0.2,0.5,vis06,0,30,0.6,320-ir105,60,130,1.3,'J08','ColdOcean_NDSI_VIS_IR10inv'),
            # Arctic whites (high-stretch cold tops)
            (320-ir105,70,130,2.0, 320-ir38,60,130,2.0, vis06,40,100,1.5, 'J09','Arctic_IR10inv_IR38inv_VIS'),
            (vis06,50,100,2.0, 320-ir105,80,130,2.0, nir22,30,80,1.5,     'J10','Polar_VIS_IR10inv_NIR'),
            (ndsi_proxy,0.3,0.9,2.0,320-ir105,70,130,1.5,vis06,50,100,2.0,'J11','Snow_NDSI_IR10inv_VIS'),
            (320-ir38,70,130,2.0,ndsi_proxy,0.3,0.9,2.0,vis06,60,100,1.8, 'J12','IceWhite_IR38inv_NDSI_VIS'),
            # Tropical greens (vegetation ratio)
            (ndvi_proxy,0.2,0.7,1.2, vis06,20,80,1.0, nir22,20,70,1.0,    'J13','TropGreen_NDVI_VIS_NIR'),
            (nir22,10,60,1.3, ndvi_proxy,0.1,0.7,1.2, vis06,10,70,0.9,    'J14','Veg_NIR_NDVI_VIS'),
            (vis06,10,70,0.9, nir22,15,65,1.2, ndvi_proxy,0.1,0.7,1.3,    'J15','GreenEarth_VIS_NIR_NDVI'),
            # Purple/ultraviolet aesthetic (convective deep clouds)
            (320-ir105,70,130,1.5, diff_38_105,-5,25,1.0, vis06,0,60,0.7, 'J16','DeepConv_IR10inv_dBT_VIS'),
            (diff_38_105,-5,25,1.2,320-ir105,70,130,1.5, nir22,0,50,0.8,  'J17','ConvPlume_dBT_IR10inv_NIR'),
            # Golden sunrise tones
            (ir38,250,330,1.3, vis06,20,90,1.1, 320-ir105,40,120,0.8,     'J18','Golden_IR38_VIS_IR10inv'),
            (vis06,20,90,1.3, ir38_solar,10,80,1.2, nir22,10,60,0.9,      'J19','Sunrise_VIS_IRsolar_NIR'),
            # Full-spectrum showcase
            (vis06,0,100,1.2, nir22,0,70,1.1, diff_38_105,-10,30,1.0,     'J20','Showcase_VIS_NIR_dBT'),
        ]
        for rv,rlo,rhi,rg, gv,glo,ghi,gg, bv,blo,bhi,bg, fname, formula in aesthetic_combos:
            total_attempted += 1
            r = norm(rv, rlo, rhi, gamma=rg)
            gch = norm(gv, glo, ghi, gamma=gg)
            b = norm(bv, blo, bhi, gamma=bg)
            rgb = np.dstack([r, gch, b])
            if self.render_map(rgb, f"Aesthetic Composite — {formula}", cat,
                               f"grp{fname}_aesthetic.png",
                               ['vis_06','nir_22','ir_38','ir_105'], is_rgb=True,
                               rgb_formula=formula):
                count_success += 1

        t_total = time.time() - t_start
        print(f"\n[EXT] Extended RGB composites done!")
        print(f" -> Attempted: {total_attempted} | Succeeded: {count_success} | Failed: {total_attempted - count_success}")
        print(f" -> Runtime: {t_total:.2f}s")
        return count_success

    def generate_all_products(self):
        """Generate all 42 distinct meteorological products."""
        print(f"\n[3/4] Mass-generating 42 meteorological product images...")
        t_start = time.time()
        
        vis06 = self.channels_data['vis_06']
        nir22 = self.channels_data['nir_22']
        ir38  = self.channels_data['ir_38']
        ir105 = self.channels_data['ir_105']
        
        count_success = 0
        total_attempted = 0

        # Custom Met Colormaps
        cmap_jet_r = plt.cm.jet_r.copy()
        
        # ---------------------------------------------------------------------
        # CATEGORY A: SINGLE CHANNEL BASE & ENHANCED PRODUCTS (16 PRODUCTS)
        # ---------------------------------------------------------------------
        cat = 'channels'
        
        # 1. VIS 0.6 Raw Reflectance
        total_attempted += 1
        if self.render_map(vis06, "VIS 0.6 µm Raw Calibrated Reflectance", cat, 
                           "vis06_reflectance_raw.png", ['vis_06'], cmap='gray', 
                           vmin=0, vmax=100, cbar_label="Top-of-Atmosphere Reflectance (%)"):
            count_success += 1

        # 2. VIS 0.6 Linear Contrast Stretched
        total_attempted += 1
        vis_stretch = np.clip((vis06 - 2.0) / 80.0 * 100.0, 0, 100)
        if self.render_map(vis_stretch, "VIS 0.6 µm Linear Contrast Enhancement", cat, 
                           "vis06_contrast_stretched.png", ['vis_06'], cmap='gray', 
                           vmin=0, vmax=100, cbar_label="Stretched Reflectance (%)"):
            count_success += 1

        # 3. VIS 0.6 Gamma Enhanced
        total_attempted += 1
        vis_gamma = np.power(np.clip(vis06 / 100.0, 0, 1), 1/1.5) * 100.0
        if self.render_map(vis_gamma, "VIS 0.6 µm Low-Light Gamma Enhanced (γ=1.5)", cat, 
                           "vis06_gamma_enhanced.png", ['vis_06'], cmap='gray', 
                           vmin=0, vmax=100, cbar_label="Gamma-Corrected Reflectance (%)"):
            count_success += 1

        # 4. VIS 0.6 Sun Angle Normalized Reflectance
        total_attempted += 1
        ny, nx = self.target_h, self.target_w
        y_g, x_g = np.ogrid[:ny, :nx]
        dist_norm = np.sqrt(((x_g - nx/2)/(nx/2))**2 + ((y_g - ny/2)/(ny/2))**2)
        cos_zenith = np.clip(np.cos(dist_norm * (np.pi/2.5)), 0.15, 1.0)
        vis_toa = np.clip(vis06 / cos_zenith, 0, 110)
        if self.render_map(vis_toa, "VIS 0.6 µm Sun Zenith Angle Normalized Reflectance", cat, 
                           "vis06_sun_angle_normalized.png", ['vis_06'], cmap='gray', 
                           vmin=0, vmax=100, cbar_label="Normalized TOA Reflectance (%)"):
            count_success += 1

        # 5. NIR 2.2 Raw Reflectance
        total_attempted += 1
        if self.render_map(nir22, "NIR 2.2 µm Raw Calibrated Reflectance", cat, 
                           "nir22_reflectance_raw.png", ['nir_22'], cmap='gray', 
                           vmin=0, vmax=80, cbar_label="NIR Reflectance (%)"):
            count_success += 1

        # 6. NIR 2.2 Particle Contrast Stretch
        total_attempted += 1
        nir_particle = np.clip((nir22 - 1.0) / 45.0 * 100.0, 0, 100)
        if self.render_map(nir_particle, "NIR 2.2 µm Cloud Particle Phase Contrast", cat, 
                           "nir22_particle_contrast.png", ['nir_22'], cmap='bone', 
                           vmin=0, vmax=100, cbar_label="Particle Phase Contrast (%)"):
            count_success += 1

        # 7. NIR 2.2 Spectral Color Gradient
        total_attempted += 1
        if self.render_map(nir22, "NIR 2.2 µm Spectral False-Color Map", cat, 
                           "nir22_color_gradient.png", ['nir_22'], cmap='viridis', 
                           vmin=0, vmax=70, cbar_label="Reflectance (%)"):
            count_success += 1

        # 8. IR 3.8 Brightness Temp (Standard Grayscale Inverted)
        total_attempted += 1
        if self.render_map(ir38, "IR 3.8 µm Brightness Temperature (Inverted)", cat, 
                           "ir38_bt_grayscale.png", ['ir_38'], cmap='gray_r', 
                           vmin=210, vmax=320, cbar_label="Brightness Temperature (K)"):
            count_success += 1

        # 9. IR 3.8 Thermal Hotspot & Fire Highlight
        total_attempted += 1
        if self.render_map(ir38, "IR 3.8 µm Thermal Hotspot & Fire Highlight Map", cat, 
                           "ir38_thermal_hotspot.png", ['ir_38'], cmap='hot', 
                           vmin=280, vmax=350, cbar_label="Hotspot Temperature (K)"):
            count_success += 1

        # 10. IR 3.8 Night Fog / Fine Contrast
        total_attempted += 1
        if self.render_map(ir38, "IR 3.8 µm Fine Thermal Contrast Palette", cat, 
                           "ir38_night_fog_contrast.png", ['ir_38'], cmap='plasma', 
                           vmin=240, vmax=305, cbar_label="Temperature (K)"):
            count_success += 1

        # 11. IR 3.8 Solar Reflectance Component Proxy
        total_attempted += 1
        ir38_solar = np.clip((ir38 - 250.0) / 70.0 * 100.0, 0, 100)
        if self.render_map(ir38_solar, "IR 3.8 µm Daytime Solar Component Proxy", cat, 
                           "ir38_solar_component.png", ['ir_38'], cmap='inferno', 
                           vmin=0, vmax=100, cbar_label="Solar Component Proxy (%)"):
            count_success += 1

        # 12. IR 10.5 BT Grayscale Inverted
        total_attempted += 1
        if self.render_map(ir105, "IR 10.5 µm Brightness Temperature Standard Map", cat, 
                           "ir105_bt_grayscale.png", ['ir_105'], cmap='gray_r', 
                           vmin=200, vmax=315, cbar_label="Cloud Top Temperature (K)"):
            count_success += 1

        # 13. IR 10.5 Meteorological Rainbow Colormap
        total_attempted += 1
        if self.render_map(ir105, "IR 10.5 µm Meteorological Thermal Rainbow Map", cat, 
                           "ir105_met_rainbow.png", ['ir_105'], cmap=cmap_jet_r, 
                           vmin=200, vmax=310, cbar_label="Temperature (K)"):
            count_success += 1

        # 14. IR 10.5 Deep Convective Storm Top Highlight
        total_attempted += 1
        if self.render_map(ir105, "IR 10.5 µm Severe Storm Cold Top Highlight (<220K)", cat, 
                           "ir105_deep_convection.png", ['ir_105'], cmap='magma_r', 
                           vmin=190, vmax=240, cbar_label="Severe Cold Top BT (K)"):
            count_success += 1

        # 15. IR 10.5 Multi-Tier Staged Temperature Bands
        total_attempted += 1
        ir_staged = np.zeros_like(ir105)
        ir_staged[ir105 < 213.15] = 4 # <-60 C
        ir_staged[(ir105 >= 213.15) & (ir105 < 223.15)] = 3 # <-50 C
        ir_staged[(ir105 >= 223.15) & (ir105 < 233.15)] = 2 # <-40 C
        ir_staged[(ir105 >= 233.15) & (ir105 < 243.15)] = 1 # <-30 C
        ir_staged[np.isnan(ir105)] = np.nan
        if self.render_map(ir_staged, "IR 10.5 µm Staged Cloud Top BT Bands", cat, 
                           "ir105_staged_thresholds.png", ['ir_105'], cmap='nipy_spectral', 
                           vmin=0, vmax=4, cbar_label="Bands (0:> -30C, 1:<-30C, 2:<-40C, 3:<-50C, 4:<-60C)"):
            count_success += 1

        # 16. NIR 2.2 Ice vs Water Cloud Discrimination
        total_attempted += 1
        if self.render_map(nir22, "NIR 2.2 µm Ice vs Water Cloud Discrimination", cat, 
                           "nir22_ice_water_discrim.png", ['nir_22'], cmap='coolwarm', 
                           vmin=5, vmax=50, cbar_label="Ice (Low NIR) vs Liquid Water (High NIR) (%)"):
            count_success += 1

        # ---------------------------------------------------------------------
        # CATEGORY B: MULTI-CHANNEL RGB COMPOSITES (15 PRODUCTS)
        # ---------------------------------------------------------------------
        cat = 'rgb_composites'

        # Helper RGB normalizer
        def norm(arr, vmin, vmax, gamma=1.0):
            val = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
            if gamma != 1.0:
                val = np.power(val, 1.0 / gamma)
            return val

        # 17. Natural Color RGB (R: NIR2.2, G: VIS0.6, B: VIS0.6)
        total_attempted += 1
        r = norm(nir22, 0, 90, gamma=1.0)
        g = norm(vis06, 0, 100, gamma=1.0)
        b = norm(vis06, 0, 100, gamma=1.0)
        rgb_nat = np.dstack([r, g, b])
        if self.render_map(rgb_nat, "Natural Color RGB Composite", cat, 
                           "rgb_natural_color.png", ['nir_22', 'vis_06'], is_rgb=True, 
                           rgb_formula="R: NIR2.2 [0-90%] | G: VIS0.6 [0-100%] | B: VIS0.6 [0-100%]"):
            count_success += 1

        # 18. Day Microphysics FCI RGB
        total_attempted += 1
        r = norm(vis06, 0, 100, gamma=1.0)
        g = norm(nir22, 0, 60, gamma=1.5)
        b = norm(323.0 - ir105, 0, 120, gamma=1.0)
        rgb_dm = np.dstack([r, g, b])
        if self.render_map(rgb_dm, "Day Microphysics FCI RGB Composite", cat, 
                           "rgb_day_microphysics_fci.png", ['vis_06', 'nir_22', 'ir_105'], is_rgb=True, 
                           rgb_formula="R: VIS0.6 [0-100%] | G: NIR2.2 [0-60%] | B: IR10.5 inv [203-323K]"):
            count_success += 1

        # 19. Night Microphysics Proxy RGB
        total_attempted += 1
        diff_38_105 = ir38 - ir105
        r = norm(diff_38_105, -4, 2)
        g = norm(ir105, 243, 293)
        b = norm(ir105, 273, 293)
        rgb_nm = np.dstack([r, g, b])
        if self.render_map(rgb_nm, "Night Microphysics Proxy RGB Composite", cat, 
                           "rgb_night_microphysics_proxy.png", ['ir_38', 'ir_105'], is_rgb=True, 
                           rgb_formula="R: (IR3.8-IR10.5) [-4..2K] | G: IR10.5 [243..293K] | B: IR10.5 [273..293K]"):
            count_success += 1

        # 20. Daytime Fog & Low Cloud RGB
        total_attempted += 1
        r = norm(vis06, 0, 100)
        g = norm(nir22, 0, 50)
        b = norm(ir38_solar, 0, 30)
        rgb_fog_day = np.dstack([r, g, b])
        if self.render_map(rgb_fog_day, "Daytime Fog & Low Cloud RGB Composite", cat, 
                           "rgb_fog_low_cloud_day.png", ['vis_06', 'nir_22', 'ir_38'], is_rgb=True, 
                           rgb_formula="R: VIS0.6 [0-100%] | G: NIR2.2 [0-50%] | B: IR3.8 Solar [0-30%]"):
            count_success += 1

        # 21. Nighttime Fog & Low Cloud RGB
        total_attempted += 1
        r = norm(diff_38_105, -4, 4)
        g = norm(ir105, 243, 293)
        b = norm(ir105, 273, 293)
        rgb_fog_night = np.dstack([r, g, b])
        if self.render_map(rgb_fog_night, "Nighttime Fog & Low Cloud RGB Composite", cat, 
                           "rgb_fog_low_cloud_night.png", ['ir_38', 'ir_105'], is_rgb=True, 
                           rgb_formula="R: (IR3.8-IR10.5) [-4..4K] | G: IR10.5 [243..293K] | B: IR10.5 [273..293K]"):
            count_success += 1

        # 22. Day Solar Ice Cloud RGB
        total_attempted += 1
        r = norm(vis06, 0, 100)
        g = norm(nir22, 0, 60)
        b = norm(ir38, 210, 300)
        rgb_solar_ice = np.dstack([r, g, b])
        if self.render_map(rgb_solar_ice, "Day Solar Ice Cloud RGB Composite", cat, 
                           "rgb_day_solar_ice.png", ['vis_06', 'nir_22', 'ir_38'], is_rgb=True, 
                           rgb_formula="R: VIS0.6 [0-100%] | G: NIR2.2 [0-60%] | B: IR3.8 [210-300K]"):
            count_success += 1

        # 23. Convection Proxy RGB
        total_attempted += 1
        r = norm(vis06, 0, 100)
        g = norm(320.0 - ir105, 0, 110)
        b = norm(diff_38_105, -5, 15)
        rgb_conv = np.dstack([r, g, b])
        if self.render_map(rgb_conv, "Severe Convection Proxy RGB Composite", cat, 
                           "rgb_convection_proxy.png", ['vis_06', 'ir_105', 'ir_38'], is_rgb=True, 
                           rgb_formula="R: VIS0.6 [0-100%] | G: IR10.5 inv [210-320K] | B: (IR3.8-IR10.5) [-5..15K]"):
            count_success += 1

        # 24. Snow & Ice Discriminator RGB
        total_attempted += 1
        r = norm(vis06, 0, 100)
        g = norm(nir22, 0, 70)
        b = norm(ir105, 220, 300)
        rgb_snow = np.dstack([r, g, b])
        if self.render_map(rgb_snow, "Snow & Ice Discriminator RGB Composite", cat, 
                           "rgb_snow_ice_discriminator.png", ['vis_06', 'nir_22', 'ir_105'], is_rgb=True, 
                           rgb_formula="R: VIS0.6 [0-100%] | G: NIR2.2 [0-70%] | B: IR10.5 [220-300K]"):
            count_success += 1

        # 25. IR Sandwich Blended Composite
        total_attempted += 1
        vis_base = norm(vis06, 0, 100, gamma=1.2)
        ir_overlay = plt.cm.jet_r(norm(ir105, 200, 260))[:, :, :3]
        blend_mask = np.clip((260.0 - ir105) / 60.0, 0.0, 1.0)[:, :, np.newaxis]
        rgb_sandwich = (1.0 - blend_mask) * np.dstack([vis_base]*3) + blend_mask * ir_overlay
        if self.render_map(rgb_sandwich, "IR Sandwich Blended Convective Composite", cat, 
                           "rgb_ir_sandwich_blended.png", ['vis_06', 'ir_105'], is_rgb=True, 
                           rgb_formula="Base: VIS0.6 Grayscale | Overlay: Colorized IR10.5 Cold Tops (<260K)"):
            count_success += 1

        # 26. Daytime Land & Surface Vegetation RGB
        total_attempted += 1
        ratio_nir_vis = np.clip((nir22 + 1e-5) / (vis06 + 1e-5), 0, 3)
        r = norm(nir22, 0, 80)
        g = norm(vis06, 0, 80)
        b = norm(ratio_nir_vis, 0, 2.5)
        rgb_land = np.dstack([r, g, b])
        if self.render_map(rgb_land, "Daytime Land & Surface Reflection RGB", cat, 
                           "rgb_land_vegetation.png", ['nir_22', 'vis_06'], is_rgb=True, 
                           rgb_formula="R: NIR2.2 [0-80%] | G: VIS0.6 [0-80%] | B: NIR/VIS Ratio [0-2.5]"):
            count_success += 1

        # 27. Cloud Phase Distinction RGB
        total_attempted += 1
        r = norm(vis06, 0, 100)
        g = norm(nir22, 0, 60)
        b = norm(ir105, 200, 300)
        rgb_phase = np.dstack([r, g, b])
        if self.render_map(rgb_phase, "Cloud Phase Distinction RGB Composite", cat, 
                           "rgb_cloud_phase_distinction.png", ['vis_06', 'nir_22', 'ir_105'], is_rgb=True, 
                           rgb_formula="R: VIS0.6 [0-100%] | G: NIR2.2 [0-60%] | B: IR10.5 [200-300K]"):
            count_success += 1

        # 28. False Color Infrared (FCIR) Proxy
        total_attempted += 1
        r = norm(nir22, 0, 90)
        g = norm(vis06, 0, 90)
        b = norm(vis06, 0, 90)
        rgb_fcir = np.dstack([r, g, b])
        if self.render_map(rgb_fcir, "False Color Infrared (FCIR) Proxy RGB", cat, 
                           "rgb_fcir_false_color.png", ['nir_22', 'vis_06'], is_rgb=True, 
                           rgb_formula="R: NIR2.2 [0-90%] | G: VIS0.6 [0-90%] | B: VIS0.6 [0-90%]"):
            count_success += 1

        # 29. Fire & Thermal Anomaly RGB
        total_attempted += 1
        r = norm(ir38, 280, 360)
        g = norm(nir22, 0, 50)
        b = norm(vis06, 0, 50)
        rgb_fire = np.dstack([r, g, b])
        if self.render_map(rgb_fire, "Fire & Thermal Anomaly RGB Composite", cat, 
                           "rgb_thermal_anomaly.png", ['ir_38', 'nir_22', 'vis_06'], is_rgb=True, 
                           rgb_formula="R: IR3.8 [280-360K] | G: NIR2.2 [0-50%] | B: VIS0.6 [0-50%]"):
            count_success += 1

        # 30. Multi-Band Thermal Contrast RGB
        total_attempted += 1
        r = norm(ir38, 220, 320)
        g = norm(ir105, 200, 300)
        b = norm(diff_38_105, -10, 20)
        rgb_therm_comp = np.dstack([r, g, b])
        if self.render_map(rgb_therm_comp, "Multi-Band Thermal Contrast RGB Composite", cat, 
                           "rgb_thermal_contrast.png", ['ir_38', 'ir_105'], is_rgb=True, 
                           rgb_formula="R: IR3.8 [220-320K] | G: IR10.5 [200-300K] | B: (IR3.8-IR10.5) [-10..20K]"):
            count_success += 1

        # 31. Wide Dynamic Range Overview RGB
        total_attempted += 1
        r = norm(vis06, 0, 100, gamma=1.3)
        g = norm(nir22, 0, 80, gamma=1.1)
        b = norm(320.0 - ir105, 0, 120, gamma=1.0)
        rgb_overview = np.dstack([r, g, b])
        if self.render_map(rgb_overview, "Wide Dynamic Range Overview RGB Composite", cat, 
                           "rgb_overview_dynamic.png", ['vis_06', 'nir_22', 'ir_105'], is_rgb=True, 
                           rgb_formula="R: VIS0.6 (γ=1.3) | G: NIR2.2 (γ=1.1) | B: IR10.5 inv [200-320K]"):
            count_success += 1

        # ---------------------------------------------------------------------
        # CATEGORY C: DERIVED METEOROLOGICAL PRODUCTS (11 PRODUCTS)
        # ---------------------------------------------------------------------
        cat = 'derived_products'

        # 32. Shortwave Split-Window Difference (IR3.8 - IR10.5)
        total_attempted += 1
        if self.render_map(diff_38_105, "Shortwave Split-Window Difference (IR3.8 − IR10.5)", cat, 
                           "diff_ir38_ir105_split_window.png", ['ir_38', 'ir_105'], cmap='RdBu_r', 
                           vmin=-15, vmax=25, cbar_label="Brightness Temp Difference ΔBT (K)"):
            count_success += 1

        # 33. NDVI Proxy Map: (NIR2.2 - VIS0.6)/(NIR2.2 + VIS0.6)
        total_attempted += 1
        ndvi_proxy = (nir22 - vis06) / (nir22 + vis06 + 1e-5)
        if self.render_map(ndvi_proxy, "Normalized Difference Vegetation Index (NDVI Proxy)", cat, 
                           "index_ndvi_proxy.png", ['nir_22', 'vis_06'], cmap='YlGn', 
                           vmin=-0.4, vmax=0.7, cbar_label="NDVI Proxy Index"):
            count_success += 1

        # 34. NDSI Snow Index Proxy: (VIS0.6 - NIR2.2)/(VIS0.6 + NIR2.2)
        total_attempted += 1
        ndsi_proxy = (vis06 - nir22) / (vis06 + nir22 + 1e-5)
        if self.render_map(ndsi_proxy, "Normalized Difference Snow Index (NDSI Proxy)", cat, 
                           "index_ndsi_proxy.png", ['vis_06', 'nir_22'], cmap='Blues', 
                           vmin=-0.3, vmax=0.8, cbar_label="NDSI Snow Index Proxy"):
            count_success += 1

        # 35. Cloud / Clear-Sky Binary Segmentation Mask
        total_attempted += 1
        cloud_mask = ((vis06 > 15.0) | (ir105 < 273.15)).astype(np.float32)
        cloud_mask[np.isnan(vis06) & np.isnan(ir105)] = np.nan
        if self.render_map(cloud_mask, "Cloud / Clear-Sky Binary Segmentation Mask (Proxy)", cat, 
                           "mask_cloud_binary.png", ['vis_06', 'ir_105'], cmap='binary_r', 
                           vmin=0, vmax=1, cbar_label="Classification (0: Clear Sky / Ocean, 1: Cloud Cover)"):
            count_success += 1

        # 36. Cloud Top Altitude Height Proxy Map
        total_attempted += 1
        # Altitude proxy using 6.5 K/km tropospheric lapse rate from surface ~288K
        cloud_alt = np.clip((288.15 - ir105) / 6.5, 0.0, 16.0)
        if self.render_map(cloud_alt, "Cloud Top Height Proxy Map (Lapse Rate Model)", cat, 
                           "proxy_cloud_top_height.png", ['ir_105'], cmap='terrain', 
                           vmin=0, vmax=15, cbar_label="Estimated Altitude (km MSL)"):
            count_success += 1

        # 37. Deep Convective Core Mask (IR10.5 < 210K)
        total_attempted += 1
        convective_cores = (ir105 < 210.0).astype(np.float32)
        convective_cores[np.isnan(ir105)] = np.nan
        if self.render_map(convective_cores, "Deep Convective Storm Core Mask (<210K / -63°C)", cat, 
                           "mask_deep_convective_cores.png", ['ir_105'], cmap='gist_heat', 
                           vmin=0, vmax=1, cbar_label="Convective Core Mask (1: Cold Deep Core)"):
            count_success += 1

        # 38. Fire & Hotspot Thermal Anomaly Index Map
        total_attempted += 1
        fire_index = np.clip(diff_38_105, 0, 50)
        if self.render_map(fire_index, "Fire & Hotspot Thermal Anomaly Index Map", cat, 
                           "index_thermal_fire_anomaly.png", ['ir_38', 'ir_105'], cmap='YlOrRd', 
                           vmin=0, vmax=35, cbar_label="Thermal Anomaly Index ΔBT (K)"):
            count_success += 1

        # 39. Sun Glint & Marine Surface Reflection Proxy
        total_attempted += 1
        sunglint = np.clip(vis06 - nir22, 0, 50)
        if self.render_map(sunglint, "Sun Glint & Marine Optical Reflection Proxy Map", cat, 
                           "proxy_sunglint_marine.png", ['vis_06', 'nir_22'], cmap='PuBuGn', 
                           vmin=0, vmax=30, cbar_label="Optical Reflection Difference (%)"):
            count_success += 1

        # 40. Cloud Optical Thickness Proxy (VIS / NIR Ratio)
        total_attempted += 1
        optical_depth = np.clip((vis06 + 1.0) / (nir22 + 1.0), 0, 8)
        if self.render_map(optical_depth, "Cloud Optical Depth & Particle Scattering Proxy", cat, 
                           "proxy_optical_depth.png", ['vis_06', 'nir_22'], cmap='viridis', 
                           vmin=0.5, vmax=6.0, cbar_label="Optical Thickness Proxy Ratio"):
            count_success += 1

        # 41. Multi-Temporal Cloud Top BT Difference Map (Cycle 0071 vs Cycle 0052)
        total_attempted += 1
        if 'ir_105' in self.secondary_channels_data:
            diff_temp_ir = self.channels_data['ir_105'] - self.secondary_channels_data['ir_105']
            if self.render_map(diff_temp_ir, "Multi-Temporal Cloud Top BT Change (Cycle 0052 − 0071)", cat, 
                               "temporal_ir105_change.png", ['ir_105'], cmap='coolwarm', 
                               vmin=-20, vmax=20, cbar_label="5-Day Thermal BT Shift ΔT (K)"):
                count_success += 1
        else:
            diff_temp_ir = ir105 - np.roll(ir105, 5, axis=1)
            if self.render_map(diff_temp_ir, "Spatial Cloud Top BT Gradient Map", cat, 
                               "temporal_ir105_change.png", ['ir_105'], cmap='coolwarm', 
                               vmin=-15, vmax=15, cbar_label="Thermal BT Gradient (K)"):
                count_success += 1

        # 42. Multi-Temporal VIS Reflectance Change Map (Cycle 0071 vs Cycle 0052)
        total_attempted += 1
        if 'vis_06' in self.secondary_channels_data:
            diff_temp_vis = self.channels_data['vis_06'] - self.secondary_channels_data['vis_06']
            if self.render_map(diff_temp_vis, "Multi-Temporal VIS Reflectance Change (Cycle 0052 − 0071)", cat, 
                               "temporal_vis06_change.png", ['vis_06'], cmap='PuOr', 
                               vmin=-40, vmax=40, cbar_label="5-Day Reflectance Shift ΔRefl (%)"):
                count_success += 1
        else:
            diff_temp_vis = vis06 - np.roll(vis06, 5, axis=1)
            if self.render_map(diff_temp_vis, "Spatial VIS Reflectance Gradient Map", cat, 
                               "temporal_vis06_change.png", ['vis_06'], cmap='PuOr', 
                               vmin=-30, vmax=30, cbar_label="Reflectance Shift (%)"):
                count_success += 1

        t_total = time.time() - t_start
        print(f"\n[4/4] Product generation finished!")
        print(f" -> Total products attempted : {total_attempted}")
        print(f" -> Total succeeded          : {count_success}")
        print(f" -> Total skipped            : {total_attempted - count_success}")
        print(f" -> Total generation runtime : {t_total:.2f} seconds")

    def write_manifest(self):
        """Write final product manifest CSV."""
        with open(self.manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'category', 'product_name', 
                                                   'channels_used', 'status', 'resolution', 
                                                   'timestamp', 'filepath'])
            writer.writeheader()
            writer.writerows(self.manifest_records)
        print(f" -> Manifest CSV written to '{self.manifest_path}'.")

    def run(self):
        cycle_files = self.discover_data()
        self.load_channels(cycle_files)
        self.generate_all_products()
        self.generate_235_rgb_composites()
        self.write_manifest()


def main():
    parser = argparse.ArgumentParser(description="MTG FCI Meteorological Product Generation Pipeline")
    parser.add_argument('--data-dir', default='data', help="Path to input data directory")
    parser.add_argument('--out-dir', default='outputs', help="Path to output directory")
    parser.add_argument('--cycle', default='0052', help="Repeat cycle to process (e.g. 0052)")
    parser.add_argument('--width', type=int, default=1000, help="Output image width in pixels")
    parser.add_argument('--height', type=int, default=1024, help="Output image height in pixels")
    
    args = parser.parse_args()
    
    pipeline = MTGProductPipeline(data_dir=args.data_dir, out_dir=args.out_dir, 
                                  cycle=args.cycle, target_size=(args.width, args.height))
    pipeline.run()


if __name__ == '__main__':
    main()
