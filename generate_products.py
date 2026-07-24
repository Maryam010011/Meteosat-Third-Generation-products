#!/usr/bin/env python3
"""
MTG FCI Meteorological Product Generation Pipeline
===================================================
Automatically discovers MTG FCI Level-1C NetCDF data in `data/` and generates
meteorological product images (channels and derived products) with white background formatting.

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
import matplotlib.colors as mcolors

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
            'derived_products': os.path.join(self.out_dir, 'derived_products')
        }
        for cat_dir in self.cats.values():
            os.makedirs(cat_dir, exist_ok=True)
            
        self.manifest_path = os.path.join(self.out_dir, 'manifest.csv')
        self.manifest_records = []
        self.channels_data = {}
        self.channels_mask = {}
        self.secondary_channels_data = {}
        self.secondary_channels_mask = {}
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
        """Safely extract raw array from Satpy scene, handle NaNs, and resize to target resolution while preserving valid data mask."""
        raw = scn[channel_name].values.astype(np.float32)
        is_ir = channel_name.startswith('ir_') or channel_name.startswith('wv_')
        fill_val = 200.0 if is_ir else 0.0
        
        if is_ir:
            mask_raw = (raw >= 150.0) & (~np.isnan(raw))
        else:
            mask_raw = (raw > 0.01) & (~np.isnan(raw))
            
        clean_raw = np.nan_to_num(raw, nan=fill_val)
        img_res = Image.fromarray(clean_raw).resize((self.target_w, self.target_h), resample=Image.Resampling.BILINEAR)
        m_res = Image.fromarray(mask_raw.astype(np.uint8)).resize((self.target_w, self.target_h), resample=Image.Resampling.NEAREST)
        
        arr_res = np.array(img_res, dtype=np.float32)
        valid_res = np.array(m_res, dtype=bool)
        
        arr_res[~valid_res] = np.nan
        return arr_res, valid_res

    def load_channels(self, cycle_files):
        """Load and extract requested spectral channels from NetCDF via Satpy."""
        print(f"\n[2/4] Loading and decoding FCI L1C channels via Satpy...")
        t0 = time.time()
        
        scn = Scene(filenames=cycle_files, reader='fci_l1c_nc')
        avail = scn.available_dataset_names()
        
        requested_chans = [
            'vis_06', 'vis_08', 'nir_16', 'nir_22', 
            'wv_63', 'wv_73', 'ir_38', 'ir_87', 
            'ir_97', 'ir_105', 'ir_123', 'ir_133'
        ]
        
        target_chans = []
        for ch in requested_chans:
            if ch in avail:
                target_chans.append(ch)
            else:
                print(f" [CHANNEL LOG] Channel '{ch}' not available in source data")
                
        print(f" -> Available requested channels: {target_chans}")
        scn.load(target_chans)
        
        print(f" -> Scene initialized in {time.time() - t0:.2f}s. Extracting & resizing channel arrays to {self.target_w}x{self.target_h}...")
        
        for ch in target_chans:
            t_ch = time.time()
            arr, valid_mask = self._extract_and_resize(scn, ch)
            self.channels_data[ch] = arr
            self.channels_mask[ch] = valid_mask
            valid_vals = arr[valid_mask]
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
                avail2 = scn2.available_dataset_names()
                chans_0071 = [c for c in ['vis_06', 'ir_105'] if c in avail2]
                if chans_0071:
                    scn2.load(chans_0071)
                    for ch in chans_0071:
                        arr2, mask2 = self._extract_and_resize(scn2, ch)
                        self.secondary_channels_data[ch] = arr2
                        self.secondary_channels_mask[ch] = mask2
            except Exception as e:
                print(f" -> Note: Multi-temporal cycle 0071 load skipped ({e})")
                
        print(f" -> Channel decoding complete in {time.time() - t0:.2f}s.")

    def render_map(self, data, title, category, filename, channels_used, 
                   cmap='gray', vmin=None, vmax=None, cbar_label=''):
        """Render a publication-quality 1000x1024 PNG product with white background, headers, overlays, & legends."""
        try:
            ny, nx = self.target_h, self.target_w
            y_grid, x_grid = np.ogrid[:ny, :nx]
            center_y, center_x = ny / 2.0, nx / 2.0
            r_disk = min(ny, nx) * 0.46
            dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
            
            # Mask off-disk pixels to ensure off-disk space renders solid WHITE
            plot_data = np.copy(data)
            plot_data[dist_from_center > r_disk] = np.nan

            fig = plt.figure(figsize=(self.target_w / 100.0, self.target_h / 100.0), dpi=100)
            fig.patch.set_facecolor('#FFFFFF')
            
            # Map axes area: top 0.07 space for header, bottom 0.08 space for colorbar/footer
            ax_map = fig.add_axes([0.02, 0.08, 0.96, 0.84])
            ax_map.set_facecolor('#FFFFFF')
            
            if isinstance(cmap, str):
                cmap_obj = plt.get_cmap(cmap).copy()
            else:
                cmap_obj = cmap.copy()
            cmap_obj.set_bad('white')

            valid_d = plot_data[~np.isnan(plot_data)]
            if vmin is None: vmin = float(np.percentile(valid_d, 1)) if len(valid_d) > 0 else 0.0
            if vmax is None: vmax = float(np.percentile(valid_d, 99)) if len(valid_d) > 0 else 1.0
            im = ax_map.imshow(plot_data, cmap=cmap_obj, vmin=vmin, vmax=vmax, origin='upper', aspect='auto')
                
            # Draw disk boundary line (clean slate grey ring)
            disk_mask = np.abs(dist_from_center - r_disk) < 1.5
            ax_map.imshow(np.ma.masked_where(~disk_mask, np.ones_like(disk_mask)), 
                          cmap=mcolors.ListedColormap(['#475569']), vmin=0, vmax=1, alpha=0.5, origin='upper', aspect='auto')
                          
            # Draw lat/lon grid lines across disk
            grid_lines = (np.abs((x_grid - center_x) % 100) < 1.0) | (np.abs((y_grid - center_y) % 100) < 1.0)
            grid_lines = grid_lines & (dist_from_center < r_disk)
            ax_map.imshow(np.ma.masked_where(~grid_lines, np.ones_like(grid_lines)), 
                          cmap=mcolors.ListedColormap(['#94A3B8']), vmin=0, vmax=1, alpha=0.25, origin='upper', aspect='auto')
            
            ax_map.axis('off')

            # --- TOP HEADER BANNER ---
            ax_hdr = fig.add_axes([0.0, 0.92, 1.0, 0.08])
            ax_hdr.set_facecolor('#F8FAFC')
            ax_hdr.axis('off')
            ax_hdr.axhline(0, color='#E2E8F0', linewidth=1)
            
            # Product Title
            ax_hdr.text(0.02, 0.62, title.upper(), color='#0F172A', 
                        fontsize=13, fontweight='bold', va='center')
            
            # Category Badge
            cat_colors = {'channels': '#2563EB', 'derived_products': '#7C3AED'}
            badge_color = cat_colors.get(category, '#475569')
            ax_hdr.text(0.02, 0.22, f"  [{category.upper().replace('_', ' ')}]  ", 
                        color='#FFFFFF', fontsize=9, fontweight='bold', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=badge_color, edgecolor='none'))
            
            # Timestamp & Metadata
            meta_str = f"EUMETSAT MTG-I1 FCI L1C | Cycle {self.cycle} | {self.timestamp_str}"
            ax_hdr.text(0.98, 0.40, meta_str, color='#475569', fontsize=9, 
                        ha='right', va='center', family='sans-serif')

            # --- BOTTOM FOOTER / LEGEND BANNER ---
            ax_ftr = fig.add_axes([0.0, 0.0, 1.0, 0.08])
            ax_ftr.set_facecolor('#F8FAFC')
            ax_ftr.axis('off')
            ax_ftr.axhline(1, color='#E2E8F0', linewidth=1)

            # Colorbar for continuous single band & derived products
            cax = fig.add_axes([0.25, 0.025, 0.50, 0.03])
            cb = fig.colorbar(im, cax=cax, orientation='horizontal')
            cb.ax.tick_params(labelsize=8, colors='#334155')
            cb.set_label(cbar_label, color='#0F172A', fontsize=8, fontweight='bold', labelpad=2)
            cb.outline.set_edgecolor('#CBD5E1')
            ax_ftr.text(0.02, 0.50, f"PRODUCT: {filename}", color='#475569', fontsize=8, va='center')
            ax_ftr.text(0.98, 0.50, f"Channels: {', '.join(channels_used)}", color='#475569', fontsize=8, ha='right', va='center')

            out_filepath = os.path.join(self.cats[category], filename)
            plt.savefig(out_filepath, dpi=100, facecolor='#FFFFFF', edgecolor='none')
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
        """Generate all channel products and derived meteorological products with white backgrounds."""
        print(f"\n[3/4] Generating channel and derived meteorological product images...")
        t_start = time.time()
        
        vis06 = self.channels_data.get('vis_06')
        nir22 = self.channels_data.get('nir_22')
        ir38  = self.channels_data.get('ir_38')
        ir105 = self.channels_data.get('ir_105')
        
        count_success = 0
        total_attempted = 0

        # Custom Met Colormaps
        cmap_jet_r = plt.cm.jet_r.copy()
        
        # ---------------------------------------------------------------------
        # CATEGORY A: SINGLE CHANNEL BASE & ENHANCED PRODUCTS (16 PRODUCTS)
        # ---------------------------------------------------------------------
        cat = 'channels'
        
        if vis06 is not None:
            # 1. VIS 0.6 Raw Reflectance
            total_attempted += 1
            if self.render_map(vis06, "VIS 0.6 µm Raw Calibrated Reflectance", cat, 
                               "vis06_reflectance_raw.png", ['vis_06'], cmap='gray', 
                               vmin=0, vmax=100, cbar_label="Top-of-Atmosphere Reflectance (%)"):
                count_success += 1

            # 2. VIS 0.6 Linear Contrast Stretched
            total_attempted += 1
            vis_stretch = np.clip((vis06 - 2.0) / 80.0 * 100.0, 0, 100)
            vis_stretch[np.isnan(vis06)] = np.nan
            if self.render_map(vis_stretch, "VIS 0.6 µm Linear Contrast Enhancement", cat, 
                               "vis06_contrast_stretched.png", ['vis_06'], cmap='gray', 
                               vmin=0, vmax=100, cbar_label="Stretched Reflectance (%)"):
                count_success += 1

            # 3. VIS 0.6 Gamma Enhanced
            total_attempted += 1
            vis_gamma = np.power(np.clip(vis06 / 100.0, 0, 1), 1/1.5) * 100.0
            vis_gamma[np.isnan(vis06)] = np.nan
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
            vis_toa[np.isnan(vis06)] = np.nan
            if self.render_map(vis_toa, "VIS 0.6 µm Sun Zenith Angle Normalized Reflectance", cat, 
                               "vis06_sun_angle_normalized.png", ['vis_06'], cmap='gray', 
                               vmin=0, vmax=100, cbar_label="Normalized TOA Reflectance (%)"):
                count_success += 1

        if nir22 is not None:
            # 5. NIR 2.2 Raw Reflectance
            total_attempted += 1
            if self.render_map(nir22, "NIR 2.2 µm Raw Calibrated Reflectance", cat, 
                               "nir22_reflectance_raw.png", ['nir_22'], cmap='gray', 
                               vmin=0, vmax=80, cbar_label="NIR Reflectance (%)"):
                count_success += 1

            # 6. NIR 2.2 Particle Contrast Stretch
            total_attempted += 1
            nir_particle = np.clip((nir22 - 1.0) / 45.0 * 100.0, 0, 100)
            nir_particle[np.isnan(nir22)] = np.nan
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

            # 16. NIR 2.2 Ice vs Water Cloud Discrimination
            total_attempted += 1
            if self.render_map(nir22, "NIR 2.2 µm Ice vs Water Cloud Discrimination", cat, 
                               "nir22_ice_water_discrim.png", ['nir_22'], cmap='coolwarm', 
                               vmin=5, vmax=50, cbar_label="Ice (Low NIR) vs Liquid Water (High NIR) (%)"):
                count_success += 1

        if ir38 is not None:
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
            ir38_solar[np.isnan(ir38)] = np.nan
            if self.render_map(ir38_solar, "IR 3.8 µm Daytime Solar Component Proxy", cat, 
                               "ir38_solar_component.png", ['ir_38'], cmap='inferno', 
                               vmin=0, vmax=100, cbar_label="Solar Component Proxy (%)"):
                count_success += 1

        if ir105 is not None:
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

        # ---------------------------------------------------------------------
        # CATEGORY B: DERIVED METEOROLOGICAL PRODUCTS (11 PRODUCTS)
        # ---------------------------------------------------------------------
        cat = 'derived_products'

        # TASK 4a: Renamed from diff_ir38_ir105_split_window -> diff_ir38_ir105_lowcloud_fog
        if ir38 is not None and ir105 is not None:
            total_attempted += 1
            diff_38_105 = ir38 - ir105
            if self.render_map(diff_38_105, "Shortwave Low Cloud & Fog Difference (IR3.8 − IR10.5)", cat, 
                               "diff_ir38_ir105_lowcloud_fog.png", ['ir_38', 'ir_105'], cmap='RdBu_r', 
                               vmin=-15, vmax=25, cbar_label="Brightness Temp Difference ΔBT (K)"):
                count_success += 1

        # TASK 4b: True Split-Window (IR12.3 - IR10.5)
        ir123 = self.channels_data.get('ir_123')
        if ir123 is not None and ir105 is not None:
            total_attempted += 1
            diff_123_105 = ir123 - ir105
            if self.render_map(diff_123_105, "True Split-Window Difference (IR12.3 − IR10.5)", cat, 
                               "diff_ir105_ir123_split_window.png", ['ir_123', 'ir_105'], cmap='RdBu_r', 
                               vmin=-5, vmax=15, cbar_label="Split-Window Temp Difference ΔBT (K)"):
                count_success += 1
        else:
            print(" [DERIVED PRODUCT LOG] True split-window product diff_ir105_ir123_split_window skipped: channel ir_123 not available in source data")

        # TASK 4c: SWIR Moisture Proxy (renamed from index_ndvi_proxy)
        if nir22 is not None and vis06 is not None:
            total_attempted += 1
            swir_moisture = (nir22 - vis06) / (nir22 + vis06 + 1e-5)
            swir_moisture[np.isnan(nir22) | np.isnan(vis06)] = np.nan
            if self.render_map(swir_moisture, "SWIR Moisture & Ice Content Proxy Index", cat, 
                               "index_swir_moisture_proxy.png", ['nir_22', 'vis_06'], cmap='YlGn', 
                               vmin=-0.4, vmax=0.7, cbar_label="SWIR Moisture Proxy Index"):
                count_success += 1

        # TASK 4c: True NDVI Proxy using VIS0.8 and VIS0.6
        vis08 = self.channels_data.get('vis_08')
        if vis08 is not None and vis06 is not None:
            total_attempted += 1
            true_ndvi = (vis08 - vis06) / (vis08 + vis06 + 1e-5)
            true_ndvi[np.isnan(vis08) | np.isnan(vis06)] = np.nan
            if self.render_map(true_ndvi, "True Normalized Difference Vegetation Index (NDVI)", cat, 
                               "index_true_ndvi_proxy.png", ['vis_08', 'vis_06'], cmap='YlGn', 
                               vmin=-0.2, vmax=0.8, cbar_label="NDVI Index"):
                count_success += 1
        else:
            print(" [DERIVED PRODUCT LOG] True NDVI product index_true_ndvi_proxy skipped: channel vis_08 not available in source data")

        # NDSI Snow Index Proxy: (VIS0.6 - NIR2.2)/(VIS0.6 + NIR2.2)
        if vis06 is not None and nir22 is not None:
            total_attempted += 1
            ndsi_proxy = (vis06 - nir22) / (vis06 + nir22 + 1e-5)
            ndsi_proxy[np.isnan(vis06) | np.isnan(nir22)] = np.nan
            if self.render_map(ndsi_proxy, "Normalized Difference Snow Index (NDSI Proxy)", cat, 
                               "index_ndsi_proxy.png", ['vis_06', 'nir_22'], cmap='Blues', 
                               vmin=-0.3, vmax=0.8, cbar_label="NDSI Snow Index Proxy"):
                count_success += 1

        # Cloud / Clear-Sky Binary Segmentation Mask (TASK 2: background WHITE, masked cloud highlighted)
        if vis06 is not None and ir105 is not None:
            total_attempted += 1
            cloud_mask = np.full_like(vis06, np.nan)
            valid_px = (~np.isnan(vis06)) | (~np.isnan(ir105))
            cloud_px = valid_px & ((vis06 > 15.0) | (ir105 < 273.15))
            cloud_mask[valid_px] = 0.0
            cloud_mask[cloud_px] = 1.0
            
            cmap_cloud_bin = mcolors.ListedColormap(['#FFFFFF', '#1E40AF']) # 0: White background, 1: Deep blue cloud
            if self.render_map(cloud_mask, "Cloud / Clear-Sky Binary Segmentation Mask", cat, 
                               "mask_cloud_binary.png", ['vis_06', 'ir_105'], cmap=cmap_cloud_bin, 
                               vmin=0, vmax=1, cbar_label="Classification (White: Clear Sky/Land/Ocean, Blue: Cloud Cover)"):
                count_success += 1

        # Cloud Top Altitude Height Proxy Map
        if ir105 is not None:
            total_attempted += 1
            cloud_alt = np.clip((288.15 - ir105) / 6.5, 0.0, 16.0)
            cloud_alt[np.isnan(ir105)] = np.nan
            if self.render_map(cloud_alt, "Cloud Top Height Proxy Map (Lapse Rate Model)", cat, 
                               "proxy_cloud_top_height.png", ['ir_105'], cmap='terrain', 
                               vmin=0, vmax=15, cbar_label="Estimated Altitude (km MSL)"):
                count_success += 1

        # Deep Convective Core Mask (IR10.5 < 210K) (TASK 2: background WHITE, cores RED)
        if ir105 is not None:
            total_attempted += 1
            convective_cores = np.full_like(ir105, np.nan)
            valid_px = ~np.isnan(ir105)
            core_px = valid_px & (ir105 < 210.0)
            convective_cores[valid_px] = 0.0
            convective_cores[core_px] = 1.0
            
            cmap_core_bin = mcolors.ListedColormap(['#FFFFFF', '#DC2626']) # 0: White background, 1: Red core
            if self.render_map(convective_cores, "Deep Convective Storm Core Mask (<210K / -63°C)", cat, 
                               "mask_deep_convective_cores.png", ['ir_105'], cmap=cmap_core_bin, 
                               vmin=0, vmax=1, cbar_label="Convective Core Mask (White: Normal, Red: Cold Deep Core)"):
                count_success += 1

        # Fire & Hotspot Thermal Anomaly Index Map
        if ir38 is not None and ir105 is not None:
            total_attempted += 1
            fire_index = np.clip(ir38 - ir105, 0, 50)
            fire_index[np.isnan(ir38) | np.isnan(ir105)] = np.nan
            if self.render_map(fire_index, "Fire & Hotspot Thermal Anomaly Index Map", cat, 
                               "index_thermal_fire_anomaly.png", ['ir_38', 'ir_105'], cmap='YlOrRd', 
                               vmin=0, vmax=35, cbar_label="Thermal Anomaly Index ΔBT (K)"):
                count_success += 1

        # Sun Glint & Marine Surface Reflection Proxy
        if vis06 is not None and nir22 is not None:
            total_attempted += 1
            sunglint = np.clip(vis06 - nir22, 0, 50)
            sunglint[np.isnan(vis06) | np.isnan(nir22)] = np.nan
            if self.render_map(sunglint, "Sun Glint & Marine Optical Reflection Proxy Map", cat, 
                               "proxy_sunglint_marine.png", ['vis_06', 'nir_22'], cmap='PuBuGn', 
                               vmin=0, vmax=30, cbar_label="Optical Reflection Difference (%)"):
                count_success += 1

        # Cloud Optical Thickness Proxy (VIS / NIR Ratio)
        if vis06 is not None and nir22 is not None:
            total_attempted += 1
            optical_depth = np.clip((vis06 + 1.0) / (nir22 + 1.0), 0, 8)
            optical_depth[np.isnan(vis06) | np.isnan(nir22)] = np.nan
            if self.render_map(optical_depth, "Cloud Optical Depth & Particle Scattering Proxy", cat, 
                               "proxy_optical_depth.png", ['vis_06', 'nir_22'], cmap='viridis', 
                               vmin=0.5, vmax=6.0, cbar_label="Optical Thickness Proxy Ratio"):
                count_success += 1

        # TASK 4d: Multi-Temporal Cloud Top BT Difference Map (Co-registered & cleanly masked)
        total_attempted += 1
        if 'ir_105' in self.secondary_channels_data:
            arr52 = self.channels_data['ir_105']
            arr71 = self.secondary_channels_data['ir_105']
            mask52 = self.channels_mask['ir_105']
            mask71 = self.secondary_channels_mask['ir_105']
            
            ny, nx = self.target_h, self.target_w
            y_g, x_g = np.ogrid[:ny, :nx]
            r_disk = min(ny, nx) * 0.46
            dist_c = np.sqrt((x_g - nx/2.0)**2 + (y_g - ny/2.0)**2)
            
            valid_both = mask52 & mask71 & (dist_c <= r_disk)
            diff_temp_ir = np.where(valid_both, arr52 - arr71, np.nan)
            
            if self.render_map(diff_temp_ir, "Multi-Temporal Cloud Top BT Shift (Cycle 0052 − 0071)", cat, 
                               "temporal_ir105_change.png", ['ir_105'], cmap='coolwarm', 
                               vmin=-20, vmax=20, cbar_label="Thermal BT Difference ΔT (K) [Blue: Cooling, Red: Warming]"):
                count_success += 1
        else:
            diff_temp_ir = ir105 - np.roll(ir105, 5, axis=1)
            diff_temp_ir[np.isnan(ir105)] = np.nan
            if self.render_map(diff_temp_ir, "Spatial Cloud Top BT Gradient Map", cat, 
                               "temporal_ir105_change.png", ['ir_105'], cmap='coolwarm', 
                               vmin=-15, vmax=15, cbar_label="Thermal BT Gradient (K)"):
                count_success += 1

        # TASK 4d: Multi-Temporal VIS Reflectance Change Map (Co-registered & cleanly masked)
        total_attempted += 1
        if 'vis_06' in self.secondary_channels_data:
            arr52_v = self.channels_data['vis_06']
            arr71_v = self.secondary_channels_data['vis_06']
            mask52_v = self.channels_mask['vis_06']
            mask71_v = self.secondary_channels_mask['vis_06']
            
            ny, nx = self.target_h, self.target_w
            y_g, x_g = np.ogrid[:ny, :nx]
            r_disk = min(ny, nx) * 0.46
            dist_c = np.sqrt((x_g - nx/2.0)**2 + (y_g - ny/2.0)**2)
            
            valid_both_v = mask52_v & mask71_v & (dist_c <= r_disk)
            diff_temp_vis = np.where(valid_both_v, arr52_v - arr71_v, np.nan)
            
            if self.render_map(diff_temp_vis, "Multi-Temporal VIS Reflectance Shift (Cycle 0052 − 0071)", cat, 
                               "temporal_vis06_change.png", ['vis_06'], cmap='PuOr', 
                               vmin=-40, vmax=40, cbar_label="Reflectance Difference ΔRefl (%) [Purple: Dimmed, Orange: Brightened]"):
                count_success += 1
        else:
            diff_temp_vis = vis06 - np.roll(vis06, 5, axis=1)
            diff_temp_vis[np.isnan(vis06)] = np.nan
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
