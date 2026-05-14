#!/usr/bin/env python3
"""
inspect_amharic_image.py – Deep inspection of a single Amharic document image.
Analyzes: dimensions, pixel stats, histogram, text lines, noise, sharpness,
binarization results, projection profiles, and more.

Usage:
    python inspect_amharic_image.py my_amharic_document.jpg [--output_dir ./inspection]
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import ndimage
from skimage import filters, morphology, measure
from collections import Counter
import json
from datetime import datetime

# -----------------------------------------------------------------------------
# Load image
# -----------------------------------------------------------------------------
def load_image(path):
    """Load image in grayscale and color."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    img_color = cv2.imread(path, cv2.IMREAD_COLOR)
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if img_gray is None:
        raise ValueError(f"Cannot read image: {path}")
    
    # Convert BGR to RGB for matplotlib
    img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB) if img_color is not None else None
    
    return img_gray, img_color_rgb


# -----------------------------------------------------------------------------
# 1. Basic Image Properties
# -----------------------------------------------------------------------------
def basic_properties(img_gray, img_color, report):
    """Print fundamental image properties."""
    report.append("## 1. Basic Image Properties\n")
    report.append("| Property | Value |")
    report.append("|----------|-------|")
    report.append(f"| Dimensions (H×W) | {img_gray.shape[0]} × {img_gray.shape[1]} |")
    report.append(f"| Total pixels | {img_gray.size:,} |")
    report.append(f"| Aspect ratio | {img_gray.shape[1]/img_gray.shape[0]:.2f} |")
    report.append(f"| Dtype | {img_gray.dtype} |")
    report.append(f"| Channels | 1 (grayscale)" + (" / 3 (color)" if img_color is not None else "") + " |")
    report.append(f"| File size (if available) | — |")
    report.append("")


# -----------------------------------------------------------------------------
# 2. Pixel Statistics
# -----------------------------------------------------------------------------
def pixel_statistics(img_gray, output_dir, report):
    """Analyze pixel intensity distribution."""
    report.append("## 2. Pixel Intensity Analysis\n")
    
    mean_val = img_gray.mean()
    std_val = img_gray.std()
    min_val = img_gray.min()
    max_val = img_gray.max()
    median_val = np.median(img_gray)
    
    report.append("| Statistic | Value |")
    report.append("|-----------|-------|")
    report.append(f"| Min | {min_val} |")
    report.append(f"| Max | {max_val} |")
    report.append(f"| Mean | {mean_val:.4f} |")
    report.append(f"| Median | {median_val:.4f} |")
    report.append(f"| Std | {std_val:.4f} |")
    report.append(f"| Dynamic range | {max_val - min_val} |")
    
    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear histogram
    axes[0].hist(img_gray.ravel(), bins=256, color='gray', edgecolor='black', alpha=0.7)
    axes[0].axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.1f}')
    axes[0].axvline(median_val, color='blue', linestyle='--', linewidth=1.5, label=f'Median: {median_val:.1f}')
    axes[0].set_title('Pixel Intensity Histogram')
    axes[0].set_xlabel('Pixel Value (0=black, 255=white)')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Log histogram
    axes[1].hist(img_gray.ravel(), bins=256, color='gray', edgecolor='black', alpha=0.7)
    axes[1].set_yscale('log')
    axes[1].axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.1f}')
    axes[1].set_title('Pixel Intensity Histogram (Log Scale)')
    axes[1].set_xlabel('Pixel Value')
    axes[1].set_ylabel('Count (log)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Image Pixel Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_pixel_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    report.append(f"\n![Pixel histogram](01_pixel_histogram.png)\n")
    
    # Threshold analysis
    thresh_otsu = filters.threshold_otsu(img_gray)
    report.append(f"\n- **Otsu threshold:** {thresh_otsu:.1f}")
    report.append(f"- **Black pixels (≤128):** {(img_gray <= 128).sum():,} ({(img_gray <= 128).mean()*100:.1f}%)")
    report.append(f"- **White pixels (>128):** {(img_gray > 128).sum():,} ({(img_gray > 128).mean()*100:.1f}%)")


# -----------------------------------------------------------------------------
# 3. Image Quality Metrics
# -----------------------------------------------------------------------------
def quality_metrics(img_gray, output_dir, report):
    """Compute sharpness, noise, contrast metrics."""
    report.append("## 3. Image Quality Metrics\n")
    
    # Sharpness (Laplacian variance)
    laplacian = ndimage.laplace(img_gray.astype(np.float64))
    sharpness = laplacian.var()
    
    # Contrast (Michelson contrast)
    min_val, max_val = img_gray.min(), img_gray.max()
    if max_val + min_val > 0:
        michelson_contrast = (max_val - min_val) / (max_val + min_val)
    else:
        michelson_contrast = 0
    
    # RMS contrast
    rms_contrast = img_gray.std() / img_gray.mean() if img_gray.mean() > 0 else 0
    
    # Noise (std after median filter)
    denoised = ndimage.median_filter(img_gray, size=3)
    noise_residual = img_gray.astype(np.float64) - denoised.astype(np.float64)
    noise_std = noise_residual.std()
    
    # Signal-to-Noise Ratio
    snr = img_gray.mean() / noise_std if noise_std > 0 else float('inf')
    
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Sharpness (Laplacian var) | {sharpness:.4f} |")
    report.append(f"| Michelson Contrast | {michelson_contrast:.4f} |")
    report.append(f"| RMS Contrast | {rms_contrast:.4f} |")
    report.append(f"| Noise Std (median residual) | {noise_std:.4f} |")
    report.append(f"| SNR (mean/noise) | {snr:.2f} dB" if snr != float('inf') else "| SNR | ∞ (no noise detected) |")
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(np.abs(laplacian), cmap='hot')
    axes[1].set_title(f'Laplacian (Edge Map)\nSharpness: {sharpness:.2f}')
    axes[1].axis('off')
    
    edges = filters.sobel(img_gray)
    axes[2].imshow(edges, cmap='hot')
    axes[2].set_title('Sobel Edges')
    axes[2].axis('off')
    
    axes[3].imshow(np.abs(noise_residual), cmap='hot')
    axes[3].set_title(f'Noise Residual\nStd: {noise_std:.3f}')
    axes[3].axis('off')
    
    plt.suptitle('Image Quality Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_quality_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    report.append(f"\n![Quality metrics](02_quality_metrics.png)\n")
    
    return sharpness, noise_std


# -----------------------------------------------------------------------------
# 4. Binarization Techniques
# -----------------------------------------------------------------------------
def binarization_analysis(img_gray, output_dir, report):
    """Test different binarization methods."""
    report.append("## 4. Binarization Analysis\n")
    
    methods = {
        'Otsu': filters.threshold_otsu(img_gray),
        'Mean': filters.threshold_mean(img_gray),
        'Li': filters.threshold_li(img_gray),
        'Yen': filters.threshold_yen(img_gray),
        'Triangle': filters.threshold_triangle(img_gray),
    }
    
    # Adaptive thresholding
    adaptive_mean = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                           cv2.THRESH_BINARY, 31, 10)
    adaptive_gaussian = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 31, 10)
    
    # Sauvola (via skimage)
    from skimage.filters import threshold_sauvola
    window_size = 51
    thresh_sauvola = threshold_sauvola(img_gray, window_size=window_size)
    binary_sauvola = img_gray > thresh_sauvola
    
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    
    # Row 1: Global methods
    row = 0
    for col, (name, thresh) in enumerate(methods.items()):
        binary = img_gray > thresh
        axes[row, col].imshow(binary, cmap='gray')
        axes[row, col].set_title(f'{name}\n(thresh={thresh:.0f})', fontsize=10)
        axes[row, col].axis('off')
    
    # Row 2: Adaptive/local methods + inverted
    row = 1
    axes[row, 0].imshow(adaptive_mean, cmap='gray')
    axes[row, 0].set_title('Adaptive Mean\n(block=31, C=10)', fontsize=10)
    axes[row, 0].axis('off')
    
    axes[row, 1].imshow(adaptive_gaussian, cmap='gray')
    axes[row, 1].set_title('Adaptive Gaussian\n(block=31, C=10)', fontsize=10)
    axes[row, 1].axis('off')
    
    axes[row, 2].imshow(binary_sauvola, cmap='gray')
    axes[row, 2].set_title(f'Sauvola\n(window={window_size})', fontsize=10)
    axes[row, 2].axis('off')
    
    # Inverted Otsu (white text on black)
    binary_otsu = img_gray > methods['Otsu']
    axes[row, 3].imshow(binary_otsu, cmap='gray')
    axes[row, 3].set_title('Otsu (black text)', fontsize=10)
    axes[row, 3].axis('off')
    
    axes[row, 4].imshow(~binary_otsu, cmap='gray')
    axes[row, 4].set_title('Otsu (white text, inverted)', fontsize=10)
    axes[row, 4].axis('off')
    
    plt.suptitle('Binarization Methods Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_binarization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    report.append(f"\n![Binarization methods](03_binarization.png)\n")
    
    # Recommend best method
    report.append("### Threshold Values\n")
    for name, thresh in methods.items():
        report.append(f"- **{name}:** {thresh:.1f}")
    report.append(f"- **Adaptive Mean:** block=31, C=10")
    report.append(f"- **Adaptive Gaussian:** block=31, C=10")
    report.append(f"- **Sauvola:** window=51")
    
    return methods


# -----------------------------------------------------------------------------
# 5. Text Region Analysis
# -----------------------------------------------------------------------------
def text_region_analysis(img_gray, output_dir, report):
    """Find text lines and connected components."""
    report.append("## 5. Text Region & Layout Analysis\n")
    
    # Binarize with Otsu
    thresh = filters.threshold_otsu(img_gray)
    binary = (img_gray > thresh).astype(np.uint8) * 255
    inverted = 255 - binary
    
    # Horizontal projection (to find text lines)
    h_proj = inverted.sum(axis=1)
    v_proj = inverted.sum(axis=0)
    
    # Find text line boundaries
    # A text line has significant ink density
    mean_h = h_proj.mean()
    threshold = mean_h * 0.3  # lines above 30% of mean
    
    line_starts = []
    line_ends = []
    in_line = False
    
    for row in range(len(h_proj)):
        if h_proj[row] > threshold and not in_line:
            line_starts.append(row)
            in_line = True
        elif h_proj[row] < threshold and in_line:
            line_ends.append(row)
            in_line = False
    if in_line:
        line_ends.append(len(h_proj) - 1)
    
    # Filter small gaps
    min_line_height = 10
    merged_starts = []
    merged_ends = []
    i = 0
    while i < len(line_starts):
        start = line_starts[i]
        end = line_ends[i]
        while i + 1 < len(line_starts) and line_starts[i+1] - line_ends[i] < min_line_height:
            end = line_ends[i+1]
            i += 1
        merged_starts.append(start)
        merged_ends.append(end)
        i += 1
    
    line_starts = merged_starts
    line_ends = merged_ends
    
    report.append(f"- **Detected text lines:** {len(line_starts)}")
    report.append(f"- **Image height:** {img_gray.shape[0]} px")
    
    if line_starts:
        heights = [line_ends[i] - line_starts[i] for i in range(len(line_starts))]
        gaps = [line_starts[i+1] - line_ends[i] for i in range(len(line_starts)-1)]
        
        report.append(f"- **Avg line height:** {np.mean(heights):.1f} px (range: {min(heights)}-{max(heights)})")
        if gaps:
            report.append(f"- **Avg line gap:** {np.mean(gaps):.1f} px (range: {min(gaps)}-{max(gaps)})")
    
    # Connected components
    labeled, num_features = measure.label(inverted, connectivity=2, return_num=True)
    props = measure.regionprops(labeled)
    
    # Filter small components (noise)
    significant = [p for p in props if p.area > 20]
    report.append(f"- **Connected components (area > 20px):** {len(significant)}")
    
    # Character size statistics
    if significant:
        widths = [p.bbox[3] - p.bbox[1] for p in significant]
        heights_cc = [p.bbox[2] - p.bbox[0] for p in significant]
        areas = [p.area for p in significant]
        
        report.append(f"- **Avg component width:** {np.mean(widths):.1f} px")
        report.append(f"- **Avg component height:** {np.mean(heights_cc):.1f} px")
        report.append(f"- **Avg component area:** {np.mean(areas):.1f} px²")
    
    # Visualize
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Original with line boxes
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.imshow(img_gray, cmap='gray')
    for start, end in zip(line_starts, line_ends):
        rect = plt.Rectangle((0, start), img_gray.shape[1], end-start, 
                              linewidth=1.5, edgecolor='red', facecolor='none', alpha=0.8)
        ax1.add_patch(rect)
    ax1.set_title(f'Detected Text Lines ({len(line_starts)})', fontsize=12)
    ax1.axis('off')
    
    # Horizontal projection
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.barh(np.arange(len(h_proj)), h_proj, height=1, color='black', alpha=0.7)
    ax2.axvline(threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold:.0f})')
    for start, end in zip(line_starts, line_ends):
        ax2.axhline(start, color='green', linestyle='-', alpha=0.5, linewidth=0.8)
        ax2.axhline(end, color='red', linestyle='-', alpha=0.5, linewidth=0.8)
    ax2.invert_yaxis()
    ax2.set_title('Horizontal Projection', fontsize=12)
    ax2.set_xlabel('Pixel Sum')
    ax2.set_ylabel('Row')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Vertical projection
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.bar(np.arange(len(v_proj)), v_proj, width=2, color='black', alpha=0.7)
    ax3.set_title('Vertical Projection', fontsize=12)
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Pixel Sum')
    ax3.grid(True, alpha=0.3)
    
    # Connected components
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.imshow(labeled, cmap='nipy_spectral')
    ax4.set_title(f'Connected Components\n({num_features} total, {len(significant)} significant)', fontsize=12)
    ax4.axis('off')
    
    plt.suptitle('Text Region & Layout Analysis', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, '04_text_regions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    report.append(f"\n![Text regions](04_text_regions.png)\n")
    
    return line_starts, line_ends


# -----------------------------------------------------------------------------
# 6. Skew Detection
# -----------------------------------------------------------------------------
def skew_detection(img_gray, output_dir, report):
    """Detect and measure image skew/rotation."""
    report.append("## 6. Skew Detection\n")
    
    # Binarize
    thresh = filters.threshold_otsu(img_gray)
    binary = (img_gray > thresh).astype(np.uint8) * 255
    inverted = 255 - binary
    
    # Use Hough transform to find line angles
    edges = cv2.Canny(inverted, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    
    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90  # Convert to angle from horizontal
            angles.append(angle)
    
    if angles:
        median_angle = np.median(angles)
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        
        report.append(f"- **Hough lines detected:** {len(angles)}")
        report.append(f"- **Median skew angle:** {median_angle:.2f}°")
        report.append(f"- **Mean skew angle:** {mean_angle:.2f}°")
        report.append(f"- **Std of angles:** {std_angle:.2f}°")
        
        if abs(median_angle) < 0.5:
            report.append("- ✅ **Image is well aligned** (no significant skew)")
        else:
            report.append(f"- ⚠️ **Image has {median_angle:.1f}° skew** — consider deskewing")
    else:
        report.append("- ⚠️ **No lines detected** — cannot estimate skew")
        median_angle = 0
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(img_gray, cmap='gray')
    if lines is not None:
        for line in lines[:50]:  # Show first 50 lines
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            axes[0].plot([x1, x2], [y1, y2], color='red', linewidth=0.5, alpha=0.5)
    axes[0].set_title('Hough Lines')
    axes[0].axis('off')
    
    if angles:
        axes[1].hist(angles, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1].axvline(median_angle, color='red', linestyle='--', linewidth=2, 
                        label=f'Median: {median_angle:.2f}°')
        axes[1].axvline(0, color='green', linestyle='-', linewidth=1, label='Horizontal (0°)')
        axes[1].set_title('Line Angle Distribution')
        axes[1].set_xlabel('Angle from Horizontal (degrees)')
        axes[1].set_ylabel('Count')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No lines detected', ha='center', va='center')
        axes[1].set_title('Line Angle Distribution')
    
    plt.suptitle('Skew Detection Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_skew_detection.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    report.append(f"\n![Skew detection](05_skew_detection.png)\n")
    
    return median_angle


# -----------------------------------------------------------------------------
# 7. OCR Readiness Check
# -----------------------------------------------------------------------------
def ocr_readiness(img_gray, sharpness, noise_std, median_skew, report):
    """Assess how suitable this image is for OCR."""
    report.append("## 7. OCR Readiness Assessment\n")
    
    score = 100
    issues = []
    
    # Check resolution
    h, w = img_gray.shape
    if h < 100 or w < 200:
        score -= 30
        issues.append("🔴 **Low resolution** — minimum 100×200 recommended for OCR")
    elif h < 300 or w < 500:
        score -= 10
        issues.append("🟡 **Moderate resolution** — OCR may be less accurate")
    else:
        issues.append("🟢 **Good resolution**")
    
    # Check contrast
    contrast = img_gray.std()
    if contrast < 30:
        score -= 20
        issues.append("🔴 **Low contrast** — text may be hard to distinguish")
    elif contrast < 50:
        score -= 10
        issues.append("🟡 **Moderate contrast**")
    else:
        issues.append("🟢 **Good contrast**")
    
    # Check sharpness
    if sharpness < 1:
        score -= 20
        issues.append("🔴 **Blurry** — edges are not sharp")
    elif sharpness < 5:
        score -= 10
        issues.append("🟡 **Slightly blurry**")
    else:
        issues.append("🟢 **Good sharpness**")
    
    # Check noise
    if noise_std > 10:
        score -= 15
        issues.append("🔴 **High noise** — may cause OCR errors")
    elif noise_std > 5:
        score -= 5
        issues.append("🟡 **Moderate noise**")
    else:
        issues.append("🟢 **Low noise**")
    
    # Check skew
    if abs(median_skew) > 2:
        score -= 20
        issues.append(f"🔴 **Significant skew ({median_skew:.1f}°)** — deskewing recommended")
    elif abs(median_skew) > 0.5:
        score -= 5
        issues.append(f"🟡 **Slight skew ({median_skew:.1f}°)**")
    else:
        issues.append("🟢 **Well aligned**")
    
    # Check if image looks like it has text
    edges_count = np.sum(filters.sobel(img_gray) > 0.05)
    if edges_count < 1000:
        score -= 30
        issues.append("🔴 **Very few edges detected** — may not contain text")
    else:
        issues.append(f"🟢 **Text-like edges detected** ({edges_count:,} edge pixels)")
    
    report.append(f"### OCR Readiness Score: **{max(0, score)}/100**\n")
    report.append("| Issue | Status |")
    report.append("|-------|--------|")
    for issue in issues:
        report.append(f"| {issue} |")
    
    report.append(f"\n**Recommendation:**")
    if score >= 80:
        report.append("- ✅ Image is well-suited for OCR. Proceed with confidence.")
    elif score >= 60:
        report.append("- 🟡 Image may benefit from preprocessing (contrast enhancement, slight sharpening).")
    else:
        report.append("- 🔴 Image needs significant preprocessing before OCR (deskewing, denoising, contrast enhancement).")
    
    return score


# -----------------------------------------------------------------------------
# 8. Model Compatibility Check
# -----------------------------------------------------------------------------
def model_compatibility(img_gray, report):
    """Check if image can be directly fed to the CRNN model."""
    report.append("## 8. Model Compatibility Check\n")
    report.append("The CRNN model expects 64×256 grayscale images.\n")
    
    h, w = img_gray.shape
    expected_h, expected_w = 64, 256
    
    report.append(f"- **Input image:** {h}×{w}")
    report.append(f"- **Model expects:** {expected_h}×{expected_w}")
    report.append(f"- **Resize needed:** YES (will be resized to {expected_h}×{expected_w})")
    
    # Aspect ratio comparison
    img_aspect = w / h
    model_aspect = expected_w / expected_h
    aspect_diff = abs(img_aspect - model_aspect) / model_aspect * 100
    
    report.append(f"- **Image aspect ratio:** {img_aspect:.3f}")
    report.append(f"- **Model aspect ratio:** {model_aspect:.3f}")
    
    if aspect_diff > 20:
        report.append(f"- ⚠️ **Aspect ratio mismatch ({aspect_diff:.0f}%)** — text may be slightly distorted after resize")
    else:
        report.append(f"- ✅ **Aspect ratio compatible** ({aspect_diff:.0f}% difference)")
    
    # Check if image is already normalized
    if img_gray.max() <= 1.0:
        report.append("- ✅ Image appears pre-normalized (values in [0,1])")
    else:
        report.append("- ℹ️ Image is 8-bit (0-255). Will be normalized during preprocessing.")


# -----------------------------------------------------------------------------
# 9. Summary Montage
# -----------------------------------------------------------------------------
def create_summary_montage(img_gray, output_dir):
    """Create a single summary image with key visualizations."""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    row = 0
    
    # Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_gray, cmap='gray')
    ax1.set_title(f'Original\n{img_gray.shape[0]}×{img_gray.shape[1]}', fontsize=10)
    ax1.axis('off')
    
    # Histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(img_gray.ravel(), bins=50, color='gray', edgecolor='black', alpha=0.7)
    ax2.set_title('Pixel Histogram', fontsize=10)
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Count')
    
    # Binarized (Otsu)
    ax3 = fig.add_subplot(gs[0, 2])
    thresh = filters.threshold_otsu(img_gray)
    ax3.imshow(img_gray > thresh, cmap='gray')
    ax3.set_title(f'Otsu Binarization\n(threshold={thresh:.0f})', fontsize=10)
    ax3.axis('off')
    
    # Edges
    ax4 = fig.add_subplot(gs[0, 3])
    edges = filters.sobel(img_gray)
    ax4.imshow(edges, cmap='hot')
    ax4.set_title('Edge Map (Sobel)', fontsize=10)
    ax4.axis('off')
    
    # Horizontal projection
    ax5 = fig.add_subplot(gs[1, :2])
    inverted = 255 - (img_gray > thresh).astype(np.uint8) * 255
    h_proj = inverted.sum(axis=1)
    ax5.barh(np.arange(len(h_proj)), h_proj, height=1, color='black', alpha=0.7)
    ax5.invert_yaxis()
    ax5.set_title('Horizontal Projection (Text Lines)', fontsize=10)
    ax5.set_xlabel('Pixel Sum')
    ax5.set_ylabel('Row')
    
    # Vertical projection
    ax6 = fig.add_subplot(gs[1, 2:])
    v_proj = inverted.sum(axis=0)
    ax6.bar(np.arange(len(v_proj)), v_proj, width=2, color='black', alpha=0.7)
    ax6.set_title('Vertical Projection (Character Spacing)', fontsize=10)
    ax6.set_xlabel('Column')
    ax6.set_ylabel('Pixel Sum')
    
    # Sharpness map
    ax7 = fig.add_subplot(gs[2, :2])
    laplacian = np.abs(ndimage.laplace(img_gray.astype(np.float64)))
    im = ax7.imshow(laplacian, cmap='hot')
    ax7.set_title(f'Sharpness Map (Laplacian)\nVariance: {laplacian.var():.2f}', fontsize=10)
    ax7.axis('off')
    plt.colorbar(im, ax=ax7, shrink=0.8)
    
    # Noise residual
    ax8 = fig.add_subplot(gs[2, 2:])
    denoised = ndimage.median_filter(img_gray, size=3)
    noise = np.abs(img_gray.astype(np.float64) - denoised.astype(np.float64))
    im2 = ax8.imshow(noise, cmap='hot')
    ax8.set_title(f'Noise Map (Median Residual)\nStd: {noise.std():.3f}', fontsize=10)
    ax8.axis('off')
    plt.colorbar(im2, ax=ax8, shrink=0.8)
    
    plt.suptitle('Amharic Document Image — Comprehensive Inspection', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(output_dir, '00_summary_montage.png'), dpi=200, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Deep inspection of an Amharic document image.")
    parser.add_argument('image', type=str, help='Path to the image file')
    parser.add_argument('--output_dir', type=str, default='./image_inspection',
                        help='Directory to save outputs')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🔍 Inspecting: {args.image}")
    print(f"{'='*60}\n")
    
    # Load
    print("Loading image...")
    img_gray, img_color = load_image(args.image)
    print(f"  Dimensions: {img_gray.shape[0]}×{img_gray.shape[1]}")
    print(f"  Dtype: {img_gray.dtype}")
    print(f"  Value range: [{img_gray.min()}, {img_gray.max()}]\n")
    
    report = []
    
    # Run all analyses
    print("Analyzing basic properties...")
    basic_properties(img_gray, img_color, report)
    
    print("Analyzing pixel statistics...")
    pixel_statistics(img_gray, args.output_dir, report)
    
    print("Computing quality metrics...")
    sharpness, noise_std = quality_metrics(img_gray, args.output_dir, report)
    
    print("Testing binarization methods...")
    binarization_analysis(img_gray, args.output_dir, report)
    
    print("Detecting text regions...")
    text_region_analysis(img_gray, args.output_dir, report)
    
    print("Detecting skew...")
    median_skew = skew_detection(img_gray, args.output_dir, report)
    
    print("Assessing OCR readiness...")
    ocr_readiness(img_gray, sharpness, noise_std, median_skew, report)
    
    print("Checking model compatibility...")
    model_compatibility(img_gray, report)
    
    print("Creating summary montage...")
    create_summary_montage(img_gray, args.output_dir)
    
    # Save report
    report_path = os.path.join(args.output_dir, 'image_inspection_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Amharic Image Inspection Report\n")
        f.write(f"**Image:** `{os.path.basename(args.image)}`\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write('\n'.join(report))
    
    print(f"\n{'='*60}")
    print(f"✅ Inspection complete!")
    print(f"📁 Output saved to: {args.output_dir}/")
    print(f"📄 Report: {report_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
