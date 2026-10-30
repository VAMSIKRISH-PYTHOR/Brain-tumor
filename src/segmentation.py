"""
Tumor Segmentation and Region of Interest (ROI) Extraction Module.

Uses:
1. Brain/head isolation
2. Adaptive high-intensity thresholding
3. Morphological operations
4. Connected-component filtering
5. ROI bounding box extraction
6. MRI visualization with tumor ROI overlay
7. Lightweight U-Net architecture for future optional use
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import FIGURES_DIR, IMAGE_SIZE, ENABLE_SEGMENTATION_ROI


def extract_roi_otsu(
    img_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], dict]:
    """
    Extract a focal ROI from an MRI image.

    The method:
    - Converts MRI to grayscale
    - Removes black background
    - Creates a brain/head mask
    - Uses adaptive intensity thresholds
    - Applies morphological cleanup
    - Selects a plausible focal connected component
    - Rejects masks that cover most of the image

    Returns:
        binary_mask:
            ROI mask, uint8 values 0 or 255

        roi_cropped:
            Cropped ROI resized back to original image dimensions

        bbox:
            (x, y, width, height)

        stats:
            Debug information
    """

    # ---------------------------------------------------------
    # 1. Validate input
    # ---------------------------------------------------------
    if img_rgb is None:
        raise ValueError("Input MRI image is None.")

    if len(img_rgb.shape) == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb.copy()

    if gray.dtype != np.uint8:
        gray = cv2.normalize(
            gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    h, w = gray.shape
    total_pixels = h * w

    # ---------------------------------------------------------
    # 2. Smooth MRI
    # ---------------------------------------------------------
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ---------------------------------------------------------
    # 3. Create brain/head mask
    #
    # IMPORTANT:
    # Do not use Otsu directly over the whole image.
    # MRI background is usually black.
    # ---------------------------------------------------------
    _, initial_brain = cv2.threshold(
        blurred,
        15,
        255,
        cv2.THRESH_BINARY
    )

    # Remove small noise
    kernel_brain = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    initial_brain = cv2.morphologyEx(
        initial_brain,
        cv2.MORPH_CLOSE,
        kernel_brain,
        iterations=3
    )

    # Keep largest connected component
    num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(
        initial_brain,
        connectivity=8
    )

    brain_mask = np.zeros_like(gray)

    if num_labels > 1:
        largest_label = 1 + np.argmax(
            stats_cc[1:, cv2.CC_STAT_AREA]
        )

        brain_mask[labels == largest_label] = 255

    brain_pixels = blurred[brain_mask > 0]

    if len(brain_pixels) == 0:
        empty_mask = np.zeros_like(gray)

        stats = {
            "input_shape": list(img_rgb.shape),
            "mask_shape": list(empty_mask.shape),
            "mask_dtype": str(empty_mask.dtype),
            "mask_min_max": [0, 0],
            "otsu_threshold_val": 0.0,
            "brain_mask_foreground_ratio": "0.00%",
            "final_roi_foreground_ratio": "0.00%",
            "nonzero_pixels": 0,
            "total_pixels": total_pixels,
            "num_connected_components": 0,
            "bounding_box": [0, 0, 0, 0],
            "fallback_method_used": "None / Empty Image",
            "is_roi_valid": False,
        }

        return (
            empty_mask,
            img_rgb.copy(),
            (0, 0, 0, 0),
            stats
        )

    # ---------------------------------------------------------
    # 4. Intracranial distance transform & skull margin safety
    # ---------------------------------------------------------
    inv_brain = cv2.bitwise_not(brain_mask)
    if np.any(inv_brain > 0):
        dist_from_bg = cv2.distanceTransform(brain_mask, cv2.DIST_L2, 5)
    else:
        border_mask = np.ones_like(gray, dtype=np.uint8)
        border_mask[1:-1, 1:-1] = 0
        dist_from_bg = cv2.distanceTransform(cv2.bitwise_not(border_mask), cv2.DIST_L2, 5)

    dist_from_bg = np.clip(dist_from_bg, 0.0, float(h // 2))
    
    # Intracranial mask: pixels at least 12 pixels INSIDE skull boundary
    _, intracranial_mask = cv2.threshold(dist_from_bg, 12, 255, cv2.THRESH_BINARY)
    intracranial_mask = intracranial_mask.astype(np.uint8)

    # Fallback to brain_mask if brain is small
    if np.sum(intracranial_mask > 0) < 0.20 * np.sum(brain_mask > 0):
        intracranial_mask = brain_mask.copy()

    brain_pixels = blurred[intracranial_mask > 0]
    if len(brain_pixels) == 0:
        brain_pixels = blurred[brain_mask > 0]
        intracranial_mask = brain_mask.copy()

    # ---------------------------------------------------------
    # 5. Calculate adaptive thresholds
    # ---------------------------------------------------------
    otsu_val, _ = cv2.threshold(
        brain_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    p90 = float(np.percentile(brain_pixels, 90))
    p93 = float(np.percentile(brain_pixels, 93))
    p95 = float(np.percentile(brain_pixels, 95))

    threshold_candidates = [
        max(float(otsu_val), p93),
        max(float(otsu_val), p90),
        p95,
    ]

    # ---------------------------------------------------------
    # 6. Generate & evaluate candidate components
    # ---------------------------------------------------------
    candidate_results = []
    candidate_telemetry = []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for thresh_idx, threshold_value in enumerate(threshold_candidates):
        _, candidate = cv2.threshold(blurred, int(threshold_value), 255, cv2.THRESH_BINARY)
        candidate = cv2.bitwise_and(candidate, intracranial_mask)

        # Remove tiny noise and bridge internal gaps
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
        candidate = cv2.dilate(candidate, kernel, iterations=1)

        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            area_ratio = area / float(total_pixels)
            x, y, bw, bh = cv2.boundingRect(contour)
            bbox_c = (x, y, bw, bh)

            aspect_ratio = max(bw / bh, bh / bw) if bh > 0 and bw > 0 else 999.0
            
            # Centroid location inside intracranial region
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else x + bw // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else y + bh // 2
            
            c_mask = np.zeros_like(gray)
            cv2.drawContours(c_mask, [contour], -1, 255, -1)
            min_skull_dist = float(np.min(dist_from_bg[c_mask > 0])) if np.any(c_mask > 0) else 0.0

            peri = cv2.arcLength(contour, True)
            circ = float((4 * np.pi * area) / (peri ** 2)) if peri > 0 else 0.0

            # Rejection Checks
            rejection_reason = None
            if area_ratio < 0.003 or area_ratio > 0.15:
                rejection_reason = f"Area out of bounds ({area_ratio*100:.2f}%)"
            elif x <= 22 or y <= 22 or x + bw >= w - 22 or y + bh >= h - 22:
                rejection_reason = f"Too close to image margin (x={x}, y={y}, w={bw}, h={bh})"
            elif bw < 10 or bh < 10:
                rejection_reason = "Dimension too small (< 10px)"
            elif aspect_ratio > 1.8:
                rejection_reason = f"Elongated strip aspect ratio ({aspect_ratio:.2f} > 1.8)"
            elif circ < 0.12:
                rejection_reason = f"Low circularity ({circ:.2f} < 0.12)"
            elif min_skull_dist < 18.0:
                rejection_reason = f"Too close to skull wall ({min_skull_dist:.1f}px < 18px)"
            elif not (0.20 * w <= cx <= 0.80 * w and 0.18 * h <= cy <= 0.70 * h):
                rejection_reason = f"Centroid ({cx}, {cy}) outside central intracranial region"
            elif y + bh > 0.80 * h:
                rejection_reason = f"Extends into lower facial/nasal region (y2={y+bh} > {int(0.80*h)})"

            if rejection_reason is None:
                mean_int = float(np.mean(blurred[c_mask > 0]))
                depth_score = float(np.mean(dist_from_bg[c_mask > 0]))
                score = float(mean_int * 0.4 + depth_score * 0.4 + circ * 50.0)

                candidate_results.append({
                    "contour": contour,
                    "area": area,
                    "area_ratio": area_ratio,
                    "bbox": bbox_c,
                    "min_skull_dist": min_skull_dist,
                    "score": score,
                    "threshold_used": threshold_value
                })
                candidate_telemetry.append({
                    "bbox": list(bbox_c),
                    "area_pixels": int(area),
                    "area_pct": f"{area_ratio*100:.2f}%",
                    "min_skull_dist": f"{min_skull_dist:.1f}px",
                    "score": f"{score:.1f}",
                    "status": "ACCEPTED"
                })
            else:
                candidate_telemetry.append({
                    "bbox": list(bbox_c),
                    "area_pixels": int(area),
                    "area_pct": f"{area_ratio*100:.2f}%",
                    "status": f"REJECTED ({rejection_reason})"
                })

    # ---------------------------------------------------------
    # 7. Select best focal ROI by candidate score
    # ---------------------------------------------------------
    mask = np.zeros_like(gray, dtype=np.uint8)
    bbox = (0, 0, 0, 0)
    roi_detected = False
    threshold_method_used = "No valid focal ROI"

    if candidate_results:
        best = max(candidate_results, key=lambda item: item["score"])
        best_contour = best["contour"]
        x, y, bw, bh = best["bbox"]
        
        cv2.drawContours(mask, [best_contour], -1, 255, -1)
        bbox = (x, y, bw, bh)
        
        roi_crop = img_rgb[y:y + bh, x:x + bw]
        if roi_crop.size > 0:
            roi_cropped = cv2.resize(roi_crop, (w, h), interpolation=cv2.INTER_LINEAR)
            roi_detected = True
            threshold_method_used = f"Intracranial Scored Focal ROI (score={best['score']:.1f}, dist={best['min_skull_dist']:.1f}px)"
        else:
            roi_cropped = img_rgb.copy()
    else:
        roi_cropped = img_rgb.copy()

    # ---------------------------------------------------------
    # 9. Final safety validation
    # ---------------------------------------------------------
    nonzero_pixels = int(
        np.sum(mask > 0)
    )

    foreground_pct = (
        nonzero_pixels / float(total_pixels)
    ) * 100.0

    # Reject obviously invalid masks.
    if (
        nonzero_pixels == 0
        or foreground_pct >= 40.0
    ):

        mask = np.zeros_like(gray)
        bbox = (0, 0, 0, 0)
        roi_cropped = img_rgb.copy()
        roi_detected = False
        nonzero_pixels = 0
        foreground_pct = 0.0

        if foreground_pct == 0:
            threshold_method_used = (
                "No valid focal lesion isolated"
            )

    # ---------------------------------------------------------
    # 10. Connected component statistics
    # ---------------------------------------------------------
    num_labels, labels, cc_stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    brain_fg_pct = (
        len(brain_pixels) / float(total_pixels)
    ) * 100.0

    stats = {
        "input_shape": list(img_rgb.shape),
        "mask_shape": list(mask.shape),
        "mask_dtype": str(mask.dtype),
        "mask_min_max": [
            int(np.min(mask)),
            int(np.max(mask))
        ],
        "otsu_threshold_val": float(otsu_val),
        "p90_threshold": float(p90),
        "p93_threshold": float(p93),
        "p95_threshold": float(p95),
        "brain_mask_foreground_ratio":
            f"{brain_fg_pct:.2f}%",
        "final_roi_foreground_ratio":
            f"{foreground_pct:.2f}%",
        "nonzero_pixels": nonzero_pixels,
        "total_pixels": total_pixels,
        "num_connected_components":
            int(max(0, num_labels - 1)),
        "bounding_box": list(bbox),
        "fallback_method_used":
            threshold_method_used,
        "candidate_telemetry": candidate_telemetry,
        "is_roi_valid": roi_detected,
    }

    return (
        mask,
        roi_cropped,
        bbox,
        stats
    )


def create_marked_mri_image(
    img_rgb: np.ndarray,
    roi_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    predicted_class: str,
    confidence: float
) -> np.ndarray:
    """
    Create annotated MRI image.

    A ROI overlay is drawn only when the segmentation
    actually produced a valid focal region.
    """

    img_marked = img_rgb.copy()

    h, w, _ = img_marked.shape

    is_tumor = (
        predicted_class.lower() != "notumor"
    )

    DISPLAY_NAMES = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "pituitary": "Pituitary Tumor",
        "notumor": "No Tumor",
    }

    display_text = DISPLAY_NAMES.get(
        predicted_class.lower(),
        predicted_class.capitalize()
    )

    has_valid_roi = (
        roi_mask is not None
        and np.any(roi_mask > 0)
        and bbox != (0, 0, 0, 0)
        and bbox[2] > 0
        and bbox[3] > 0
        and bbox[2] < w
        and bbox[3] < h
    )

    # ---------------------------------------------------------
    # Tumor + valid ROI
    # ---------------------------------------------------------
    if is_tumor and has_valid_roi:

        # Red transparent overlay
        overlay = img_marked.copy()

        overlay[roi_mask > 0] = [
            255,
            60,
            60
        ]

        cv2.addWeighted(
            overlay,
            0.40,
            img_marked,
            0.60,
            0,
            img_marked
        )

        # Tumor contour
        contours, _ = cv2.findContours(
            roi_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            img_marked,
            contours,
            -1,
            (255, 40, 40),
            2
        )

        # Bounding box
        x, y, bw, bh = bbox

        cv2.rectangle(
            img_marked,
            (x, y),
            (x + bw, y + bh),
            (255, 220, 0),
            2
        )

        label_text = (
            f"MODEL PREDICTION: "
            f"{display_text.upper()} "
            f"({confidence:.1f}%)"
        )

        banner_bg = (
            200,
            30,
            30
        )

    # ---------------------------------------------------------
    # Tumor but no valid ROI
    # ---------------------------------------------------------
    elif is_tumor:

        label_text = (
            f"MODEL PREDICTION: "
            f"{display_text.upper()} "
            f"({confidence:.1f}%)"
        )

        banner_bg = (
            200,
            30,
            30
        )

    # ---------------------------------------------------------
    # No tumor
    # ---------------------------------------------------------
    else:

        label_text = (
            f"MODEL PREDICTION: "
            f"{display_text.upper()} "
            f"({confidence:.1f}%)"
        )

        banner_bg = (
            30,
            160,
            30
        )

    # ---------------------------------------------------------
    # Diagnostic banner
    # ---------------------------------------------------------
    banner_height = 36

    banner = np.zeros(
        (banner_height, w, 3),
        dtype=np.uint8
    )

    banner[:] = banner_bg

    # Make text size fit
    font_scale = 0.45

    cv2.putText(
        banner,
        label_text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    marked_result = np.vstack(
        [
            banner,
            img_marked
        ]
    )

    return marked_result


# =============================================================
# Lightweight U-Net
# =============================================================

class DoubleConv(nn.Module):
    """
    Convolution -> BatchNorm -> ReLU
    repeated twice.
    """

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )

    def forward(self, x):
        return self.conv(x)


class LightweightUNet(nn.Module):
    """
    Lightweight U-Net architecture.

    Input:
        (B, 3, H, W)

    Output:
        (B, 1, H, W)
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1
    ):
        super().__init__()

        self.inc = DoubleConv(
            in_channels,
            32
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(32, 64)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.conv_up1 = DoubleConv(
            128,
            64
        )

        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.conv_up2 = DoubleConv(
            64,
            32
        )

        self.outc = nn.Conv2d(
            32,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x = self.up1(x3)

        x = torch.cat(
            [x, x2],
            dim=1
        )

        x = self.conv_up1(x)

        x = self.up2(x)

        x = torch.cat(
            [x, x1],
            dim=1
        )

        x = self.conv_up2(x)

        logits = self.outc(x)

        return torch.sigmoid(logits)


# =============================================================
# Pipeline entry point
# =============================================================

def apply_segmentation_pipeline(
    img_rgb: np.ndarray,
    enable_roi: bool = ENABLE_SEGMENTATION_ROI
) -> np.ndarray:
    """
    Pipeline entry point.

    Returns ROI-cropped image when enabled.
    Otherwise returns original image.
    """

    if not enable_roi:
        return img_rgb

    _, roi_cropped, _, stats = extract_roi_otsu(
        img_rgb
    )

    # If segmentation failed, don't destroy the input image.
    if not stats["is_roi_valid"]:
        return img_rgb

    return roi_cropped


# =============================================================
# Test
# =============================================================

def main():

    print(
        "Testing Stage 3: "
        "Tumor Segmentation & ROI Extractor..."
    )

    processed_dir = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "processed"
    )

    sample_files = list(
        processed_dir.rglob("*.png")
    )

    if not sample_files:

        print(
            "[WARNING] No preprocessed images "
            "found for segmentation demo."
        )

        return

    print(
        f"Loaded sample image: "
        f"{sample_files[0].name}"
    )

    img_bgr = cv2.imread(
        str(sample_files[0])
    )

    if img_bgr is None:

        print(
            "[ERROR] Could not read image."
        )

        return

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    # ---------------------------------------------------------
    # Extract ROI
    # ---------------------------------------------------------
    (
        mask,
        roi_cropped,
        bbox,
        stats
    ) = extract_roi_otsu(
        img_rgb
    )

    print("\n========== ROI DEBUG ==========")

    for key, value in stats.items():
        print(f"{key}: {value}")

    print("===============================\n")

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    axes[0].imshow(
        img_rgb
    )

    axes[0].set_title(
        "Preprocessed MRI"
    )

    axes[0].axis("off")

    axes[1].imshow(
        mask,
        cmap="gray"
    )

    axes[1].set_title(
        "Morphological Candidate ROI"
    )

    axes[1].axis("off")

    axes[2].imshow(
        roi_cropped
    )

    axes[2].set_title(
        f"ROI Crop\nbbox={bbox}"
    )

    axes[2].axis("off")

    plt.tight_layout()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    out_fig_path = (
        FIGURES_DIR
        / "segmentation_demo.png"
    )

    plt.savefig(
        out_fig_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Segmentation figure saved to: "
        f"{out_fig_path}"
    )

    # ---------------------------------------------------------
    # Test U-Net
    # ---------------------------------------------------------
    unet = LightweightUNet()

    dummy_input = torch.randn(
        1,
        3,
        224,
        224
    )

    dummy_output = unet(
        dummy_input
    )

    print(
        "U-Net verified."
    )

    print(
        f"Input: {dummy_input.shape}"
    )

    print(
        f"Output: {dummy_output.shape}"
    )

    print(
        "Stage 3 completed successfully!"
    )


if __name__ == "__main__":
    main()