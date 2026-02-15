"""
Evaluate and Visualize EnMAP+S2 Super-Resolution Results
Combined script for computing metrics (Ds, Dλ, QNR) and creating visualizations
"""

import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom, convolve, laplace
import rasterio
from rasterio.transform import Affine
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


def create_msi_reference_geotiff(mat_path, original_s2_path, output_path, ratio=3, ms_key="I_MS"):
    """Create I_MS crop tiff from original S2 tile (for reference)"""
    dat = sio.loadmat(mat_path)
    I_MS = dat[ms_key]
    H, W, n_bands = I_MS.shape
    
    with rasterio.open(original_s2_path) as src:
        s2_crs = src.crs
        s2_transform = src.transform
        s2_height = src.height
        s2_width = src.width
    
    row0_s2 = (s2_height - H) // 2
    col0_s2 = (s2_width - W) // 2
    crop_transform = s2_transform * Affine.translation(col0_s2, row0_s2)
    
    I_MS_transposed = np.transpose(I_MS, (2, 0, 1)).astype(np.float32)
    
    with rasterio.open(
        output_path, 'w',
        driver='GTiff', height=H, width=W, count=n_bands,
        dtype=rasterio.float32, crs=s2_crs, transform=crop_transform,
        compress='lzw', tiled=True, blockxsize=256, blockysize=256
    ) as dst:
        dst.write(I_MS_transposed)
    
    print(f"  Created reference: {output_path}")
    return str(output_path), s2_crs, crop_transform


def crop_with_transform(img, transform, border):
    """Crop image borders and update affine transform accordingly."""
    if border is None or border <= 0:
        return img, transform

    if img.ndim == 3:
        img_c = img[border:-border, border:-border, :]
    elif img.ndim == 2:
        img_c = img[border:-border, border:-border]
    else:
        raise ValueError("Unsupported image dimensions")

    new_transform = transform * Affine.translation(border, border)
    return img_c, new_transform


def crop_border(arr, border=3):
    """Crop border pixels from 2D or 3D arrays."""
    if border is None or border <= 0:
        return arr
    if arr.ndim == 2:
        return arr[border:-border, border:-border]
    if arr.ndim == 3:
        return arr[border:-border, border:-border, :]
    raise ValueError(f"Unsupported ndim={arr.ndim} for border crop")


def _normalize_per_band(img, eps=1e-8):
    """Normalize to [0,1] per band (independent scaling)."""
    img = img.astype(np.float64)
    if img.ndim == 2:
        m, M = img.min(), img.max()
        return (img - m) / (M - m + eps)
    if img.ndim == 3:
        out = np.empty_like(img, dtype=np.float64)
        for b in range(img.shape[2]):
            m, M = img[..., b].min(), img[..., b].max()
            out[..., b] = (img[..., b] - m) / (M - m + eps)
        return out
    raise ValueError("img must be 2D or 3D")


def scale_ms_f_to_ms(I_MS, I_MS_f, eps=1e-8):
    """Shared scaling: normalize BOTH using min/max from I_MS per band."""
    I_MS = I_MS.astype(np.float64)
    I_MS_f = I_MS_f.astype(np.float64)
    out_ms = np.empty_like(I_MS, dtype=np.float64)
    out_f = np.empty_like(I_MS_f, dtype=np.float64)

    for b in range(I_MS.shape[2]):
        mn = I_MS[..., b].min()
        mx = I_MS[..., b].max()
        denom = (mx - mn + eps)
        out_ms[..., b] = (I_MS[..., b] - mn) / denom
        out_f[..., b] = (I_MS_f[..., b] - mn) / denom

    return out_ms, out_f


def uiqi_blockwise(img1, img2, block_size=32, eps=1e-12, clip=True):
    """Universal Image Quality Index (UIQI) computed blockwise."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    H, W = img1.shape

    q_vals = []
    for i in range(0, H - block_size + 1, block_size):
        for j in range(0, W - block_size + 1, block_size):
            b1 = img1[i:i+block_size, j:j+block_size].ravel()
            b2 = img2[i:i+block_size, j:j+block_size].ravel()

            mu1 = b1.mean()
            mu2 = b2.mean()

            v1 = b1.var(ddof=0)
            v2 = b2.var(ddof=0)
            cov = ((b1 - mu1) * (b2 - mu2)).mean()

            num = 4.0 * cov * mu1 * mu2
            den = (v1 + v2 + eps) * (mu1**2 + mu2**2 + eps)
            q = num / den
            if clip:
                q = float(np.clip(q, -1.0, 1.0))
            q_vals.append(q)

    return float(np.mean(q_vals)) if q_vals else 0.0


def _highpass(img):
    """Simple 3x3 Laplacian-like high-pass filter."""
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]], dtype=np.float64) / 4.0
    return convolve(img, kernel, mode="nearest")


def _upsample_hsi(I_HS, target_shape):
    """Bicubic upsampling of LR-HSI to match SR-HSI spatial size."""
    h, w, _ = I_HS.shape
    H, W = target_shape
    scale_h, scale_w = H / h, W / w
    return zoom(I_HS, (scale_h, scale_w, 1), order=3)


def _fix_srf_shape(R, n_ms_bands, n_hs_bands):
    """Robustly fix SRF matrix to correct shape (n_ms_bands, n_hs_bands)."""
    if R is None:
        return None

    print(f"  Fixing SRF shape: input shape {R.shape}")
    print(f"  Expected: ({n_ms_bands}, {n_hs_bands})")

    R = np.squeeze(R)

    if R.ndim != 2:
        raise ValueError(f"SRF must be 2D, got shape {R.shape}")

    r0, r1 = R.shape

    if r0 == n_ms_bands and r1 == n_hs_bands:
        print(f"  SRF shape is correct: ({r0}, {r1})")
        return R
    elif r0 == n_hs_bands and r1 == n_ms_bands:
        print(f"  SRF is transposed! {R.shape} -> {R.T.shape}")
        return R.T
    else:
        raise ValueError(
            f"SRF shape {R.shape} incompatible with "
            f"MS bands={n_ms_bands}, HS bands={n_hs_bands}"
        )


def _hs_to_ms(I_HS, R, output_dir, border_crop, export_geotiffs, crs, transform):
    """Synthesize MSI from HSI using spectral response matrix R."""
    H, W, L = I_HS.shape
    l, Lr = R.shape

    if L != Lr:
        raise ValueError(f"SRF shape ({l},{Lr}) incompatible with HSI bands {L}")

    I_MS_synth = np.einsum('hwk,mk->hwm', I_HS, R)
    
    # Export synthetic MSI if GeoTIFF export enabled
    if export_geotiffs and output_dir is not None:
        _write_geotiff(
            I_MS_synth, "synthetic_MSI.tif", output_dir, 
            border_crop, crs, transform
        )
    
    return I_MS_synth


def _write_geotiff(img, filename, output_dir, border_crop, crs, transform):
    """Write GeoTIFF with border cropping."""
    img_c, transform_c = crop_with_transform(img, transform, border_crop)

    H, W, B = img_c.shape
    img_transposed = np.transpose(img_c, (2, 0, 1)).astype(np.float32)

    suffix = f"_buffer{border_crop}" if border_crop > 0 else ""
    filename = filename.replace(".tif", f"{suffix}.tif")
    path = output_dir / filename

    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=H, width=W,
        count=B,
        dtype=rasterio.float32,
        crs=crs,
        transform=transform_c,
        compress='lzw',
        tiled=True,
        blockxsize=256,
        blockysize=256
    ) as dst:
        dst.write(img_transposed)

    print(f"  Saved: {filename} ({H}x{W}x{B})")
    return str(path)


def compute_D_lambda(I_HS, I_SR, num_bands=None, block_size=32, border_crop=3):
    """Spectral distortion index Dλ."""
    H, W, L = I_SR.shape
    h, w, Lh = I_HS.shape
    assert L == Lh, "I_HS and I_SR must have same number of bands"

    I_HS_up = _upsample_hsi(I_HS, (H, W))

    I_HS_up = crop_border(I_HS_up, border_crop)
    I_SR_c = crop_border(I_SR, border_crop)

    I_HS_up = _normalize_per_band(I_HS_up)
    I_SR_c = _normalize_per_band(I_SR_c)

    if num_bands is None or num_bands > L:
        num_bands = L

    band_idx = np.linspace(0, L - 1, num_bands, dtype=int)

    q_hs_pairs = []
    q_sr_pairs = []

    for i in range(len(band_idx)):
        for j in range(i + 1, len(band_idx)):
            b1, b2 = band_idx[i], band_idx[j]

            q_hs = uiqi_blockwise(I_HS_up[..., b1], I_HS_up[..., b2],
                                  block_size=block_size)
            q_sr = uiqi_blockwise(I_SR_c[..., b1], I_SR_c[..., b2],
                                  block_size=block_size)

            q_hs_pairs.append(q_hs)
            q_sr_pairs.append(q_sr)

    D_lambda = float(np.mean(np.abs(np.array(q_sr_pairs) - np.array(q_hs_pairs))))
    return D_lambda


def compute_D_s(I_MS, I_SR, R, block_size, border_crop, 
                export_geotiffs, output_dir, crs, transform):
    """Spatial distortion index Ds."""
    H, W, l = I_MS.shape
    H2, W2, L = I_SR.shape
    assert (H, W) == (H2, W2), "Spatial shapes of I_MS and I_SR must match"

    # Synthesize MSI from fused HSI
    if R is not None:
        I_MS_f = _hs_to_ms(I_SR, R, output_dir, border_crop, 
                          export_geotiffs, crs, transform)
    else:
        band_idx = np.linspace(0, L - 1, l, dtype=int)
        I_MS_f = I_SR[..., band_idx]
        if export_geotiffs and output_dir is not None:
            _write_geotiff(
                I_MS_f, "synthetic_MSI.tif", output_dir,
                border_crop, crs, transform
            )

    I_MS_c = crop_border(I_MS, border_crop)
    I_MS_f_c = crop_border(I_MS_f, border_crop)

    I_MS_n, I_MS_f_n = scale_ms_f_to_ms(I_MS_c, I_MS_f_c)

    I_MS_int = I_MS_n.mean(axis=2)
    I_MS_f_int = I_MS_f_n.mean(axis=2)

    HP_MS = _highpass(I_MS_int)
    HP_MS_f = _highpass(I_MS_f_int)

    q = uiqi_blockwise(HP_MS, HP_MS_f, block_size=block_size)
    D_s = float(np.abs(q - 1.0))
    return D_s


def compute_QNR(D_lambda, D_s, alpha=1.0, beta=1.0):
    """Overall Quality with No Reference (QNR)."""
    return float((1.0 - D_lambda) ** alpha * (1.0 - D_s) ** beta)


def create_rgb(img, bands=[50, 30, 20], p_low=1, p_high=99):
    """Create RGB composite from hyperspectral (or multispectral) image."""
    H, W, B = img.shape
    r = min(bands[0], B - 1)
    g = min(bands[1], B - 1)
    b = min(bands[2], B - 1)

    rgb = np.stack([img[:, :, r], img[:, :, g], img[:, :, b]], axis=2)

    vmin, vmax = np.percentile(rgb, [p_low, p_high])
    rgb = np.clip((rgb - vmin) / (vmax - vmin + 1e-8), 0, 1)

    return rgb


def create_visualizations(I_HS, I_MS, X, output_dir, ratio):
    """Create comprehensive visualization plots."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Row 1: Full RGB views
    ax1 = fig.add_subplot(gs[0, 0])
    rgb_lr = create_rgb(I_HS)
    ax1.imshow(rgb_lr)
    ax1.set_title(
        f'Input: LR-HSI (EnMAP)\n{I_HS.shape[0]}x{I_HS.shape[1]}x{I_HS.shape[2]}\n60m resolution',
        fontsize=12, fontweight='bold', color='blue'
    )
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    rgb_msi = create_rgb(I_MS, bands=[2, 1, 0], p_low=1, p_high=99)
    ax2.imshow(rgb_msi)
    ax2.set_title(
        f'Input: HR-MSI (Sentinel-2)\n{I_MS.shape[0]}x{I_MS.shape[1]}x{I_MS.shape[2]}\n10m resolution',
        fontsize=12, fontweight='bold', color='green'
    )
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    rgb_sr = create_rgb(X, p_low=1, p_high=99)
    ax3.imshow(rgb_sr)
    ax3.set_title(
        f'Output: Super-Resolved HSI\n{X.shape[0]}x{X.shape[1]}x{X.shape[2]}\n10m resolution',
        fontsize=12, fontweight='bold', color='red'
    )
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    ax4.text(
        0.5, 0.5, "Full Views",
        transform=ax4.transAxes,
        ha='center', va='center',
        fontsize=14, fontweight='bold'
    )

    # Row 2: Zoomed views
    h_start, h_end = X.shape[0] // 4, 3 * X.shape[0] // 4
    w_start, w_end = X.shape[1] // 4, 3 * X.shape[1] // 4

    lr_h_start = h_start // ratio
    lr_h_end = h_end // ratio
    lr_w_start = w_start // ratio
    lr_w_end = w_end // ratio

    I_HS_zoom = I_HS[lr_h_start:lr_h_end, lr_w_start:lr_w_end, :]
    rgb_lr_zoom = create_rgb(I_HS_zoom)
    rgb_lr_zoom_up = zoom(rgb_lr_zoom, (ratio, ratio, 1), order=1)

    rgb_msi_zoom = rgb_msi[h_start:h_end, w_start:w_end, :]
    rgb_sr_zoom = rgb_sr[h_start:h_end, w_start:w_end, :]

    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(rgb_lr_zoom_up)
    ax5.set_title(
        'LR-HSI (EnMAP) Zoom\nUpsampled to 10m',
        fontsize=12, fontweight='bold', color='blue'
    )
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(rgb_msi_zoom)
    ax6.set_title(
        'HR-MSI (S2) Zoom\nCenter Region',
        fontsize=12, fontweight='bold', color='green'
    )
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(rgb_sr_zoom)
    ax7.set_title(
        'Super-Resolved HSI Zoom\nCenter Region',
        fontsize=12, fontweight='bold', color='red'
    )
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    ax8.text(
        0.5, 0.5, f"Zoomed comparison\n({ratio*10}m -> 10m)",
        transform=ax8.transAxes,
        ha='center', va='center',
        fontsize=12, fontweight='bold'
    )

    # Row 3: Spectral signatures & stats
    h_c, w_c = X.shape[0] // 2, X.shape[1] // 2

    ax9 = fig.add_subplot(gs[2, 0:2])
    hs_row = h_c // ratio
    hs_col = w_c // ratio
    ax9.plot(I_HS[hs_row, hs_col, :],
             'b-', linewidth=2, alpha=0.7, label='LR-HSI (60m)')
    ax9.plot(X[h_c, w_c, :],
             'r-', linewidth=2, label='Super-Resolved (10m)')
    ax9.set_xlabel('Band Index', fontsize=11)
    ax9.set_ylabel('Reflectance', fontsize=11)
    ax9.set_title('Spectral Signature near Center', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=10)
    ax9.grid(True, alpha=0.3)
    ax9.set_ylim([0, 1])

    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.axis('off')
    stats_text = f"""
SUPER-RESOLUTION RESULTS
{'='*31}

Input Images:
  - LR-HSI: {I_HS.shape[0]}x{I_HS.shape[1]}x{I_HS.shape[2]} (60m, EnMAP)
  - HR-MSI: {I_MS.shape[0]}x{I_MS.shape[1]}x{I_MS.shape[2]} (10m, S2)

Output:
  - SR-HSI: {X.shape[0]}x{X.shape[1]}x{X.shape[2]} (10m)
  
Enhancement:
  - Spatial: {ratio}x upsampling (60m -> 10m)
  - Spectral: {X.shape[2]} bands maintained
  - Resolution: {X.shape[0]*X.shape[1]:,} pixels

Data Range (SR-HSI):
  - Min: {X.min():.4f}
  - Max: {X.max():.4f}
  - Mean: {X.mean():.4f}
"""
    ax10.text(
        0.1, 0.9, stats_text, transform=ax10.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )

    fig.suptitle(
        'EnMAP + Sentinel-2 Super-Resolution Results (SDP Method)',
        fontsize=16, fontweight='bold', y=0.98
    )

    viz_path = output_dir / "visualization.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {viz_path}")
    plt.close()

    # Save RGB composite
    rgb_output = (rgb_sr * 255).astype(np.uint8)
    rgb_path = output_dir / "super_resolved_rgb.png"
    Image.fromarray(rgb_output).save(rgb_path)
    print(f"RGB composite saved to: {rgb_path}")

    # Save specific bands
    for band_idx in [20, 30, 50, 70]:
        if band_idx < X.shape[2]:
            band_img = (X[:, :, band_idx] * 255).astype(np.uint8)
            band_path = output_dir / f"super_resolved_band_{band_idx:03d}.png"
            Image.fromarray(band_img).save(band_path)
            print(f"Band {band_idx} saved")

    return str(viz_path), str(rgb_path)


def evaluate_and_visualize(args):
    """Main evaluation and visualization function."""
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*72)
    print("LOADING DATA")
    print("="*72)
    
    # Load HSI & MSI
    dat = sio.loadmat(args.data_mat)
    I_HS = dat['I_HS']
    I_MS = dat['I_MS']
    
    print(f"  Loaded I_HS: {I_HS.shape}")
    print(f"  Loaded I_MS: {I_MS.shape}")
    
    # Load SRF R
    R = None
    if args.srf_mat is not None:
        rdat = sio.loadmat(args.srf_mat)
        if "R" in rdat:
            R = rdat["R"]
            print(f"  Loaded R from {args.srf_mat}: shape {R.shape}")
    elif "R" in dat:
        R = dat["R"]
        print(f"  Loaded R from data file: shape {R.shape}")
    
    # Fix SRF shape if needed
    if R is not None:
        R = _fix_srf_shape(R, I_MS.shape[2], I_HS.shape[2])
    
    # Load fused SR-HSI
    X_mat = sio.loadmat(args.sr_mat)
    X = X_mat['X']
    print(f"  Loaded X: {X.shape}")
    X = np.squeeze(X, axis=0)
    I_SR = np.transpose(X, (1, 2, 0))
    print(f"  Reshaped to I_SR: {I_SR.shape}")
    
    # Infer spatial ratio
    ratio_h = I_SR.shape[0] // I_HS.shape[0]
    ratio_w = I_SR.shape[1] // I_HS.shape[1]
    ratio = int((ratio_h + ratio_w) / 2)
    print(f"  Estimated spatial ratio: {ratio}")
    
    # Setup GeoTIFF metadata if export enabled
    crs = None
    transform = None
    
    if args.export_geotiffs:
        print("\n" + "="*72)
        print("LOADING GEOSPATIAL METADATA")
        print("="*72)
        
        if args.msi_crop_tif and Path(args.msi_crop_tif).exists():
            print(f"  Using existing reference: {args.msi_crop_tif}")
            with rasterio.open(args.msi_crop_tif) as src:
                crs = src.crs
                transform = src.transform
        elif args.original_s2_tif:
            print(f"  Creating I_MS reference from original S2 tile...")
            msi_crop_tif_path = output_dir / "I_MS_crop_reference.tif"
            _, crs, transform = create_msi_reference_geotiff(
                args.data_mat, args.original_s2_tif, msi_crop_tif_path, 3, "I_MS"
            )
        else:
            raise ValueError(
                "Need either --msi-crop-tif or --original-s2-tif for GeoTIFF export!"
            )
        
        print(f"    CRS: {crs}")
        print(f"    Transform: {transform}")
    
    # Export SR-HSI as GeoTIFF if enabled
    if args.export_geotiffs:
        print("\n" + "="*72)
        print("EXPORTING GEOTIFF FILES")
        print("="*72)
        _write_geotiff(I_SR, "SR_HSI.tif", output_dir, 
                      args.border_crop, crs, transform)
    
    # Compute metrics
    print("\n" + "="*72)
    print("COMPUTING METRICS")
    print("="*72)
    
    print(f"  Computing D_lambda (spectral distortion)...")
    D_lambda = compute_D_lambda(
        I_HS, I_SR,
        num_bands=args.num_bands_D_lambda,
        block_size=args.block_size,
        border_crop=args.border_crop
    )
    print(f"  D_lambda = {D_lambda:.4f}")
    
    print(f"\n  Computing D_s (spatial distortion)...")
    D_s = compute_D_s(
        I_MS, I_SR, R,
        block_size=args.block_size,
        border_crop=args.border_crop,
        export_geotiffs=args.export_geotiffs,
        output_dir=output_dir,
        crs=crs,
        transform=transform
    )
    print(f"  D_s = {D_s:.4f}")
    
    print(f"\n  Computing QNR (overall quality)...")
    QNR = compute_QNR(D_lambda, D_s)
    print(f"  QNR = {QNR:.4f}")
    
    # Print results
    print("\n" + "="*72)
    print("RESULTS")
    print("="*72)
    print(f"  D_lambda (spectral): {D_lambda:.4f}  (0 = best)")
    print(f"  D_s (spatial):       {D_s:.4f}  (0 = best)")
    print(f"  QNR (overall):       {QNR:.4f}  (1 = best)")
    
    # Create visualizations
    print("\n" + "="*72)
    print("CREATING VISUALIZATIONS")
    print("="*72)
    
    viz_path, rgb_path = create_visualizations(
        I_HS, I_MS, I_SR, output_dir, ratio
    )
    
    print("\n" + "="*72)
    print("EVALUATION COMPLETE")
    print("="*72)
    
    return {"D_lambda": D_lambda, "D_s": D_s, "QNR": QNR}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate and visualize EnMAP+S2 super-resolution results'
    )
    
    # Required arguments
    parser.add_argument(
        '--data-mat',
        type=str,
        required=True,
        help='Path to input data MAT file (contains I_HS, I_MS, optionally R)'
    )
    parser.add_argument(
        '--sr-mat',
        type=str,
        required=True,
        help='Path to super-resolved result MAT file (contains X)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory for saving outputs'
    )
    
    # Optional arguments
    parser.add_argument(
        '--srf-mat',
        type=str,
        default=None,
        help='Path to SRF MAT file (if not in data-mat)'
    )
    parser.add_argument(
        '--original-s2-tif',
        type=str,
        default=None,
        help='Path to original S2 tile for GeoTIFF metadata'
    )
    parser.add_argument(
        '--msi-crop-tif',
        type=str,
        default=None,
        help='Path to existing I_MS reference GeoTIFF'
    )
    parser.add_argument(
        '--export-geotiffs',
        action='store_true',
        help='Export results as GeoTIFF files'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=16,
        help='Block size for UIQI computation. Default: 16'
    )
    parser.add_argument(
        '--border-crop',
        type=int,
        default=1,
        help='Border pixels to crop for metrics. Default: 1'
    )
    parser.add_argument(
        '--num-bands-D-lambda',
        type=int,
        default=50,
        help='Number of bands for D_lambda computation. Default: 50'
    )
    
    args = parser.parse_args()
    
    # Run evaluation and visualization
    metrics = evaluate_and_visualize(args)
    
    print(f"\nFinal metrics:")
    print(f"  D_lambda = {metrics['D_lambda']:.4f}")
    print(f"  D_s      = {metrics['D_s']:.4f}")
    print(f"  QNR      = {metrics['QNR']:.4f}")
