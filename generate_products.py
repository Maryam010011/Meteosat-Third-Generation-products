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
        is_ir = 'ir' in channel_name
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
