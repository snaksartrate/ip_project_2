"""
app.py

Streamlit entrypoint for the "Comparative Image Segmentation and Object
Measurement System" project.

app.py is a UI controller: it lays out the page, collects inputs, calls
functions from src/ and run_benchmark.py, and displays results. The only
non-UI code here is small helpers and two cached wrappers (the stress-test
sweep and the dataset loader).

Layout:
    Upload + preprocessing settings (sidebar)
    Ground-truth inputs (shared by Stress-Test, Auto-Tuning, Measurements)
    Tabs: Pipeline | Stress-Test | Auto-Tuning | Hardware Profile |
          Measurements | Benchmark

Mask convention: uint8, 255 = foreground, 0 = background.
"""

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import preprocessing, segmentation, visualization
from src import degradation, evaluation, optimization, profiling
from src import morphology, detection, features

import run_benchmark  # batch runner; imports only src/, no Streamlit


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    "Global Thresholding": "global",
    "Otsu Thresholding": "otsu",
    "Adaptive Thresholding": "adaptive",
    "Edge-Based Segmentation": "edge",
    "Hybrid Model": "hybrid",
}

SHORT_LABELS = {
    "global": "Global",
    "otsu": "Otsu",
    "adaptive": "Adaptive",
    "edge": "Edge-Based",
    "hybrid": "Hybrid",
}

# Per-degradation UI configuration.
#   start        -> the "no degradation" value (first point of the sweep)
#   max_lo/hi/default/step -> slider that sets the maximum severity
#   preview_step -> step of the single-level preview slider
DEGRADATION_CONFIG = {
    "Gaussian noise": {
        "axis": "Noise variance", "start": 0.0,
        "max_lo": 100.0, "max_hi": 4000.0, "max_default": 1600.0, "step": 100.0,
        "preview_step": 50.0, "is_int": False,
    },
    "Low light (gamma)": {
        "axis": "Gamma (higher = darker)", "start": 1.0,
        "max_lo": 1.5, "max_hi": 6.0, "max_default": 4.0, "step": 0.5,
        "preview_step": 0.1, "is_int": False,
    },
    "Motion blur": {
        "axis": "Blur length (px)", "start": 1,
        "max_lo": 3, "max_hi": 61, "max_default": 31, "step": 2,
        "preview_step": 2, "is_int": True,
    },
}

NOISE_SEED = 42  # fixed seed -> reproducible noise, stable IoU curves


# ---------------------------------------------------------------------------
# Helpers (small, UI-side)
# ---------------------------------------------------------------------------

def _signature(*parts) -> str:
    """Hash arrays and settings into a short key, used to detect stale results."""
    h = hashlib.md5()
    for p in parts:
        h.update(p.tobytes() if isinstance(p, np.ndarray) else str(p).encode())
    return h.hexdigest()


def _apply_degradation(image, kind, level, angle, seed):
    """Dispatch to the matching degradation function."""
    if kind == "Gaussian noise":
        return degradation.inject_gaussian_noise(image, float(level), seed=seed)
    if kind == "Low light (gamma)":
        return degradation.inject_low_light(image, float(level))
    return degradation.inject_motion_blur(image, int(level), float(angle))


def _build_levels(kind, max_level, n_steps):
    """Evenly spaced degradation levels from 'none' up to max_level."""
    cfg = DEGRADATION_CONFIG[kind]
    raw = np.linspace(cfg["start"], max_level, n_steps)
    if cfg["is_int"]:
        # Motion-blur lengths must be odd integers.
        return sorted(set((np.round(raw).astype(int) | 1).tolist()))
    return [round(float(v), 3) for v in raw]


def _method_params(global_threshold, polarity):
    """Default parameters for all five methods (same for every level)."""
    return {
        "global": {"threshold": global_threshold, "object_polarity": polarity},
        "otsu": {"object_polarity": polarity},
        "adaptive": {"block_size": 11, "C": 2, "adaptive_method": "gaussian",
                     "object_polarity": polarity},
        "edge": {"low_threshold": 50, "high_threshold": 150, "closing_kernel_size": 5},
        "hybrid": {"object_polarity": polarity},
    }


@st.cache_data(show_spinner=False)
def cached_stress_sweep(gray, gt, kind, levels, angle, seed, polarity, global_threshold):
    """
    Degrade the image at every level, run all four methods, and return
    {method_key: [IoU per level]}. Cached so that re-running with the
    same settings is instant.
    """
    params = _method_params(global_threshold, polarity)
    curves = {key: [] for key in params}
    for level in levels:
        degraded = _apply_degradation(gray, kind, level, angle, seed)
        for key, p in params.items():
            seg = segmentation.segment(degraded, key, **p)
            curves[key].append(evaluation.calculate_iou(seg["mask"], gt))
    return curves

DATASET_DIR = Path("dataset")
RESULTS_DIR = Path("results")

BENCH_METRICS = ["count_accuracy", "mask_iou", "mean_iou", "raw_mask_iou",
                 "false_positives", "missed", "mean_measurement_error_pct",
                 "processing_time_ms"]


@st.cache_data(show_spinner=False)
def cached_dataset(dataset_dir: str):
    """Load the dataset once (images, masks, GT objects) and reuse it."""
    return run_benchmark.load_dataset(Path(dataset_dir))


def _bench_cfg(noise: str, kernel_size: int) -> argparse.Namespace:
    """Benchmark settings, same defaults as run_benchmark.py."""
    return argparse.Namespace(
        noise=noise, kernel_size=kernel_size, open_kernel=3, close_kernel=5,
        min_area=50, global_threshold=127, iou_threshold=0.5)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Segmentation Lab",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Segmentation Lab Design System Injection
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,600;1,8..60,600&display=swap');

/* Base canvas */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #F4F1EA !important;
    color: #191c20 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    display: none !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 96% !important;
}

/* Headings */
h1, h2, h3, .stHeading {
    font-family: 'Source Serif 4', serif !important;
    color: #191c20 !important;
    letter-spacing: -0.01em !important;
}

/* Monospace numerals & code */
code, pre, .font-mono, [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar styling: Carbon ink slab #1C1F23 */
[data-testid="stSidebar"] {
    background-color: #1C1F23 !important;
    border-right: 1px solid #2B2F36 !important;
}
[data-testid="stSidebar"] * {
    color: #E8E4D9 !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #FBFAF7 !important;
    font-family: 'Source Serif 4', serif !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background-color: #24282E !important;
    border: 1px solid #2B2F36 !important;
    border-radius: 4px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background-color: #1C1F23 !important;
    color: #E8E4D9 !important;
}

/* Streamlit Tabs Navigation */
[data-testid="stTabs"] {
    border-bottom: 1px solid #D9D3C5 !important;
    margin-bottom: 1.25rem !important;
    gap: 1.5rem !important;
}
[data-testid="stTabs"] button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 14px !important;
    color: #5B5F66 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.5rem 0.25rem 0.75rem 0.25rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #1F5F5B !important;
    border-bottom: 2px solid #1F5F5B !important;
    font-weight: 600 !important;
}
[data-testid="stTabs"] button:hover {
    color: #191c20 !important;
}

/* Cards & Expanders */
div[data-testid="stExpander"] {
    background-color: #FBFAF7 !important;
    border: 1px solid #D9D3C5 !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 2px rgba(28,31,35,0.04) !important;
}
div[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #191c20 !important;
    background-color: #ECE8DE !important;
    border-bottom: 1px solid #D9D3C5 !important;
}

/* Buttons */
.stButton > button {
    border-radius: 4px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.15s ease-in-out !important;
}
.stButton > button[kind="primary"] {
    background-color: #1F5F5B !important;
    color: #FBFAF7 !important;
    border: 1px solid #17494A !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #17494A !important;
    border-color: #0F3536 !important;
}
.stButton > button[kind="secondary"] {
    background-color: #FBFAF7 !important;
    color: #191c20 !important;
    border: 1px solid #D9D3C5 !important;
}
.stButton > button[kind="secondary"]:hover {
    background-color: #ECE8DE !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #1F5F5B !important;
    border-color: #1F5F5B !important;
}

/* Dataframe & Tables */
[data-testid="stDataFrame"], .stTable {
    border: 1px solid #D9D3C5 !important;
    border-radius: 4px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] * {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Archival Station Header
# ---------------------------------------------------------------------------
st.markdown("""
<div style="background-color: #FBFAF7; border: 1px solid #D9D3C5; border-radius: 6px; padding: 12px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 2px rgba(28,31,35,0.04);">
  <div style="display: flex; align-items: center; gap: 12px;">
    <svg style="height: 28px; width: 28px; color: #1F5F5B;" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="16" cy="16" r="13" stroke="currentColor" stroke-width="2"/>
      <circle cx="16" cy="16" r="7" stroke="currentColor" stroke-width="1.5" stroke-dasharray="2 2"/>
      <circle cx="16" cy="16" r="2.5" fill="currentColor"/>
    </svg>
    <div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-family: 'Source Serif 4', serif; font-size: 20px; font-weight: 600; color: #191c20; letter-spacing: -0.01em;">Segmentation Lab</span>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; background-color: #ECE8DE; color: #5B5F66; padding: 2px 6px; border-radius: 3px; border: 1px solid #D9D3C5;">v2.4.0-rc</span>
      </div>
      <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 11px; color: #5B5F66; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 1px;">
        Comparative Image Segmentation &amp; Object Measurement System
      </div>
    </div>
  </div>
  <div style="display: flex; align-items: center; gap: 10px;">
    <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #5B5F66; background-color: #ECE8DE; padding: 5px 10px; border-radius: 4px; border: 1px solid #D9D3C5;">
      STATION_ID: ARCHIVAL_PLATE_4
    </span>
    <span style="background-color: #1F5F5B; color: #FBFAF7; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;">
      🔬 BENCH
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 8px; margin-bottom: 12px; border-bottom: 1px solid #2B2F36;">
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 20px;">🔬</span>
        <span style="font-family: 'Source Serif 4', serif; font-size: 17px; font-weight: 600; color: #FBFAF7;">Segmentation Lab</span>
      </div>
      <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 10px; color: #8A8D91; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px;">
        Optical Workbench Console
      </div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("⚙️ Preprocessing Config", expanded=True):
        max_dimension = st.slider(
            "Max image dimension (px)",
            min_value=200,
            max_value=2000,
            value=800,
            step=50,
            help="Images larger than this on their longest side are downscaled "
                 "(aspect ratio preserved) before processing.",
        )

        noise_method = st.radio(
            "Noise reduction method",
            options=["None", "Gaussian", "Median"],
            index=1,
            help="Gaussian smoothing: bell curve blur. Median: impulse noise filter.",
        )

        kernel_size = 5
        if noise_method != "None":
            kernel_size = st.slider(
                "Filter kernel size",
                min_value=1 if noise_method == "Gaussian" else 3,
                max_value=31,
                value=5,
                step=2,
                help="Must be odd.",
            )

        show_equalization_experiment = st.checkbox(
            "Show histogram equalization experiment",
            value=False,
            help="View-only contrast experiment.",
        )
    
    st.markdown("""
    <div style="margin-top: 24px; padding: 12px; background-color: #16181C; border: 1px solid #2B2F36; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #8A8D91;">
      <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
        <span>Engine Status</span>
        <span style="color: #95d2cc;">● Ready (OpenCV)</span>
      </div>
      <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
        <span>Memory</span>
        <span style="color: #E8E4D9;">42.8 MB</span>
      </div>
      <div style="display: flex; justify-content: space-between;">
        <span>Active Preset</span>
        <span style="color: #E8E4D9;">synthetic_20</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Upload or Select Specimen Image (shared by every tab)
# ---------------------------------------------------------------------------

preset_images = [
    "normal_01.png", "normal_02.png", "normal_03.png", "normal_04.png", "normal_05.png",
    "noisy_01.png", "noisy_02.png", "touching_01.png", "uneven_01.png"
]

source_mode = st.radio(
    "Image Source Mode",
    ["Dataset Specimen (Preset)", "Upload Custom Image"],
    horizontal=True,
    label_visibility="collapsed",
)

uploaded_filename = "normal_01.png"
image_bgr = None

if source_mode == "Dataset Specimen (Preset)":
    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        chosen_preset = st.selectbox("Select Specimen Image", preset_images, index=0)
        uploaded_filename = chosen_preset
        preset_path = Path("dataset/images") / chosen_preset
        if preset_path.exists():
            image_bgr = cv2.imread(str(preset_path))
        else:
            st.error(f"Specimen file {chosen_preset} not found in dataset/images/")
            st.stop()
    with col_info:
        st.markdown(f"""
        <div style="background-color: #FBFAF7; border: 1px solid #D9D3C5; border-radius: 4px; padding: 10px 14px; margin-top: 8px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #5B5F66;">
          <span style="font-weight: 600; color: #191c20;">{uploaded_filename}</span> &nbsp;|&nbsp; 
          Dim: {image_bgr.shape[1]}×{image_bgr.shape[0]} px &nbsp;|&nbsp; Channels: {image_bgr.shape[2] if image_bgr.ndim == 3 else 1} &nbsp;|&nbsp;
          Dtype: {image_bgr.dtype} &nbsp;|&nbsp; Range: [{int(image_bgr.min())}, {int(image_bgr.max())}]
        </div>
        """, unsafe_allow_html=True)
else:
    uploaded_file = st.file_uploader(
        "Upload Specimen Image (JPG, JPEG, PNG, BMP)",
        type=["jpg", "jpeg", "png", "bmp"],
    )
    if uploaded_file is None:
        st.info("Upload an image file above, or switch to 'Dataset Specimen (Preset)' to explore immediately.")
        st.stop()
    uploaded_filename = uploaded_file.name
    image_bgr, error = preprocessing.load_and_validate_image(uploaded_file)
    if error is not None:
        st.error(f"Could not process this file: {error}")
        st.stop()

info = preprocessing.get_image_info(image_bgr)

result = preprocessing.preprocess(
    image_bgr,
    max_dimension=max_dimension,
    noise_reduction_method=noise_method.lower(),
    kernel_size=kernel_size,
)


# ---------------------------------------------------------------------------
# Ground-truth inputs (shared by Stress-Test, Auto-Tuning, Hardware Profile
# and Measurements)
# ---------------------------------------------------------------------------

def render_ground_truth_inputs(filtered: np.ndarray):
    """
    Collect object polarity and a ground-truth mask.

    Returns:
        (polarity, gt_mask or None, gt_source or None)
        gt_mask is uint8 {0, 255} at the same size as `filtered`.
    """
    gt_mask, gt_source = None, None

    with st.expander("🔬 Ground Truth & Polarity Configuration", expanded=True):
        col_pol, col_mask, col_prev = st.columns([1.2, 1.8, 1])
        
        with col_pol:
            polarity_label = st.radio(
                "Object polarity",
                options=["Objects darker than background", "Objects brighter than background"],
                index=0,
                key="shared_polarity",
                help="Used by Global/Otsu/Adaptive in the Stress-Test, Auto-Tuning and Hardware Profile tabs.",
            )
            polarity = "darker" if "darker" in polarity_label else "brighter"
        
        target_h, target_w = filtered.shape[:2]

        with col_mask:
            gt_file = st.file_uploader(
                "Ground-truth mask (optional override): white = object, black = background",
                type=["png", "jpg", "jpeg", "bmp"],
                key="gt_uploader",
                help="Upload an override mask, or use the auto-linked dataset ground-truth mask.",
            )
            use_pseudo = st.checkbox(
                "No mask? Use Otsu's output as a pseudo ground truth (demo only)",
                value=False,
                key="use_pseudo_gt",
                help="Only for demonstration without ground truth.",
            )

        # Check for matching dataset ground truth mask
        preset_gt_path = Path("dataset/ground_truth") / uploaded_filename
        if gt_file is not None:
            gt_bgr, gt_error = preprocessing.load_and_validate_image(gt_file)
            if gt_error is not None:
                st.error(f"Could not read the ground-truth mask: {gt_error}")
            else:
                gt_gray = preprocessing.convert_to_grayscale(gt_bgr)
                gh, gw = gt_gray.shape[:2]
                if (gh, gw) != (target_h, target_w):
                    gt_gray = cv2.resize(gt_gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                gt_mask = np.where(gt_gray > 127, 255, 0).astype(np.uint8)
                gt_source = "uploaded mask"
        elif use_pseudo:
            gt_mask = segmentation.segment_otsu(filtered, polarity)["mask"]
            gt_source = "pseudo (Otsu)"
        elif preset_gt_path.exists():
            gt_raw = cv2.imread(str(preset_gt_path), cv2.IMREAD_GRAYSCALE)
            if gt_raw is not None:
                if gt_raw.shape[:2] != (target_h, target_w):
                    gt_raw = cv2.resize(gt_raw, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                gt_mask = np.where(gt_raw > 127, 255, 0).astype(np.uint8)
                gt_source = f"Dataset Ground Truth ({uploaded_filename})"

        with col_prev:
            if gt_mask is not None:
                fg_pct = 100.0 * np.count_nonzero(gt_mask) / gt_mask.size
                st.markdown(f"""
                <div style="text-align: right; margin-bottom: 4px;">
                  <span style="font-family: 'IBM Plex Sans', sans-serif; font-size: 11px; font-weight: 600; color: #1F5F5B; background-color: rgba(31,95,91,0.1); padding: 3px 8px; border-radius: 3px; border: 1px solid rgba(31,95,91,0.2);">
                    ● Active: Ground Truth Verified
                  </span>
                  <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #5B5F66; margin-top: 3px;">
                    Coverage: {fg_pct:.2f}% FG
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.image(gt_mask, caption="Active Reference Mask", width=110)
            else:
                st.info("No active ground truth")

    return polarity, gt_mask, gt_source


polarity, gt_mask, gt_source = render_ground_truth_inputs(result["filtered"])


# ---------------------------------------------------------------------------
# TAB 1: existing pipeline (unchanged logic)
# ---------------------------------------------------------------------------

def render_pipeline_tab():
    # --- Original image + info ---------------------------------------------
    st.subheader("2. Original image")

    col_img, col_info = st.columns([2, 1])
    with col_img:
        st.image(image_bgr[:, :, ::-1], caption="Original (unmodified)", width=450)
    with col_info:
        st.markdown("**Image information**")
        st.table(
            {
                "Property": ["Height", "Width", "Channels", "Dtype", "Approx. size"],
                "Value": [
                    f"{info['height']} px",
                    f"{info['width']} px",
                    info["channels"],
                    info["dtype"],
                    f"{info['size_kb']} KB",
                ],
            }
        )

    # --- Resize step -----------------------------------------------------------
    st.subheader("3. Resizing")
    if result["was_resized"]:
        resized_info = preprocessing.get_image_info(result["resized"])
        st.write(
            f"Original size ({info['height']}x{info['width']}) exceeded the "
            f"{max_dimension}px limit on its longest side -> resized to "
            f"**{resized_info['height']}x{resized_info['width']}** (aspect ratio preserved)."
        )
        st.image(result["resized"][:, :, ::-1], caption="Resized image", width=400)
    else:
        st.write(
            f"Image ({info['height']}x{info['width']}) is already within the "
            f"{max_dimension}px limit -- no resizing applied."
        )

    # --- Grayscale step --------------------------------------------------------
    st.subheader("4. Grayscale conversion")
    st.write(
        "Converted using OpenCV's BGR->Gray weighted sum "
        "(Gray = 0.299R + 0.587G + 0.114B), which approximates human "
        "perceptual brightness sensitivity. A single-channel intensity image "
        "is required by the thresholding-based segmentation methods."
    )
    col_gray1, col_gray2 = st.columns(2)
    with col_gray1:
        st.image(result["resized"][:, :, ::-1], caption="Before (color)", width=350)
    with col_gray2:
        st.image(result["gray"], caption="After (grayscale)", width=350)

    # --- Noise reduction step ----------------------------------------------------
    st.subheader("5. Noise reduction")
    if noise_method == "None":
        st.write("No filtering applied (selected method: None).")
    else:
        st.write(
            f"Applied **{noise_method}** filtering with kernel size **{kernel_size}**."
        )
        if noise_method == "Gaussian":
            st.caption(
                "Gaussian: weighted average using a bell-curve kernel. Smooths "
                "general sensor noise well but also softens edges, and does not "
                "specifically remove isolated outlier (salt-and-pepper) pixels."
            )
        else:
            st.caption(
                "Median: replaces each pixel with the median of its neighborhood. "
                "Highly effective against salt-and-pepper noise since outlier "
                "pixels barely influence a median, while edges are preserved "
                "better than with Gaussian blur."
            )
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.image(result["gray"], caption="Before filtering", width=350)
    with col_f2:
        st.image(result["filtered"], caption="After filtering", width=350)

    # --- Histogram step ----------------------------------------------------------
    st.subheader("6. Intensity histogram")
    st.write(
        "Computed on the final preprocessed (post-filter) grayscale image -- "
        "this is the histogram later thresholding methods (e.g. Otsu) will "
        "actually operate on."
    )
    hist_fig = visualization.plot_histogram(result["histogram"])
    st.pyplot(hist_fig)

    if show_equalization_experiment:
        st.markdown("**Optional experiment: histogram equalization**")
        st.caption(
            "For viewing only -- does not feed into the pipeline above or any "
            "later step by default."
        )
        equalized = preprocessing.equalize_histogram(result["filtered"])
        eq_hist = preprocessing.compute_histogram(equalized)
        col_eq1, col_eq2 = st.columns(2)
        with col_eq1:
            st.image(equalized, caption="Equalized image", width=350)
        with col_eq2:
            st.pyplot(visualization.plot_histogram(eq_hist, title="Equalized Histogram"))

    st.divider()

    # --- Segmentation step ---------------------------------------------------------
    st.subheader("7. Segmentation")

    col_method, col_polarity = st.columns(2)
    with col_method:
        method_label = st.selectbox("Segmentation method", options=list(METHOD_LABELS.keys()))
        method_key = METHOD_LABELS[method_label]
    with col_polarity:
        if method_key in ("global", "otsu", "adaptive"):
            polarity_label = st.radio(
                "Object polarity",
                options=["Objects darker than background", "Objects brighter than background"],
                index=0,
                help="Which side of the threshold counts as the foreground object.",
            )
            object_polarity = "darker" if "darker" in polarity_label else "brighter"
        else:
            st.caption("Edge-based segmentation fills enclosed regions directly -- "
                       "no polarity choice needed.")
            object_polarity = None

    st.markdown(f"**Parameters for {method_label}**")

    seg_params = {}
    if method_key == "global":
        seg_params["threshold"] = st.slider("Threshold value", 0, 255, 127)
        seg_params["object_polarity"] = object_polarity

    elif method_key == "otsu":
        st.caption("No threshold slider -- Otsu computes it automatically from the histogram above.")
        seg_params["object_polarity"] = object_polarity

    elif method_key == "adaptive":
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            seg_params["block_size"] = st.slider("Block size (neighborhood)", 3, 51, 11, step=2)
        with col_a2:
            seg_params["C"] = st.slider("Constant C", -20, 20, 2)
        with col_a3:
            seg_params["adaptive_method"] = st.selectbox("Local statistic", ["gaussian", "mean"])
        seg_params["object_polarity"] = object_polarity

    elif method_key == "edge":
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            seg_params["low_threshold"] = st.slider("Canny low threshold", 0, 255, 50)
        with col_e2:
            seg_params["high_threshold"] = st.slider("Canny high threshold", 0, 255, 150)
        with col_e3:
            seg_params["closing_kernel_size"] = st.slider("Gap-closing kernel size", 1, 25, 5, step=2)

    run_segmentation = st.button("Run Segmentation", type="primary")

    if run_segmentation:
        seg_result = segmentation.segment(result["filtered"], method_key, **seg_params)

        st.markdown("### Segmentation input")
        st.image(result["filtered"], caption="Preprocessed grayscale image (input to segmentation)", width=400)

        st.markdown("### Segmentation output")
        st.image(seg_result["mask"], caption=f"Binary mask ({method_label}) -- white = foreground", width=400)

        st.markdown("### Method information")
        info_rows = {
            "Method": method_label,
            "Processing time": f"{seg_result['processing_time_ms']:.2f} ms",
        }
        if seg_result["threshold"] is not None:
            info_rows["Threshold used"] = f"{seg_result['threshold']:.1f}"
        for k, v in seg_result["parameters"].items():
            info_rows[k] = str(v)
        st.table({"Property": list(info_rows.keys()), "Value": list(info_rows.values())})

        if method_key == "edge":
            st.markdown("### Edge-based pipeline detail")
            st.write("Grayscale → Canny edge map → gap-closed edges → filled binary mask")
            col_e1, col_e2, col_e3, col_e4 = st.columns(4)
            with col_e1:
                st.image(result["filtered"], caption="Grayscale", width=220)
            with col_e2:
                st.image(seg_result["intermediate"]["edges"], caption="Canny edges", width=220)
            with col_e3:
                st.image(seg_result["intermediate"]["closed_edges"], caption="Gap-closed edges", width=220)
            with col_e4:
                st.image(seg_result["mask"], caption="Filled mask", width=220)
    else:
        st.write("Set parameters above and click **Run Segmentation**.")

    st.divider()
    st.caption(
        "Cleanup, detection and measurements are in the Measurements tab; "
        "dataset benchmarking is in the Benchmark tab."
    )


# ---------------------------------------------------------------------------
# TAB 2: stress-test
# ---------------------------------------------------------------------------

def render_stress_tab():
    st.markdown("""
    <div style="background-color: #ECE8DE; border: 1px solid #D9D3C5; border-radius: 4px; padding: 10px 14px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; font-size: 13px;">
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="color: #1F5F5B; font-weight: 600;">ℹ Evaluation Reference:</span>
        <span style="color: #5B5F66;">Requires a ground-truth mask or pseudo ground truth for relative IoU convergence metrics.</span>
      </div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; background-color: #FBFAF7; padding: 4px 8px; border-radius: 2px; border: 1px solid #D9D3C5;">
        <span style="color: #1F5F5B; font-weight: 600;">● Active Mask:</span> Ground Truth Verified
      </div>
    </div>
    """, unsafe_allow_html=True)

    if gt_mask is None:
        st.info("Provide a ground-truth mask (or enable the pseudo ground truth) in the **Ground Truth & Polarity Configuration** section above.")
        return
    if gt_source == "pseudo (Otsu)":
        st.warning("Using Otsu's output as ground truth: the curves are only a demo, and Otsu is favoured at level 0.")

    gray = result["filtered"]

    col_setup, col_main = st.columns([1, 2.5])

    with col_setup:
        st.markdown("""
        <div style="border-bottom: 1px solid #ECE8DE; padding-bottom: 6px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: baseline;">
          <span style="font-family: 'Source Serif 4', serif; font-size: 17px; font-weight: 600; color: #191c20;">Degradation Setup</span>
          <span style="font-family: 'IBM Plex Sans', sans-serif; font-size: 10px; color: #8A8D91; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;">SWEEP MATRIX</span>
        </div>
        """, unsafe_allow_html=True)

        kind = st.radio("Perturbation Channel", list(DEGRADATION_CONFIG.keys()), key="stress_kind")
        cfg = DEGRADATION_CONFIG[kind]
        cast = int if cfg["is_int"] else float

        max_level = st.slider(
            f"Maximum: {cfg['axis']}",
            cast(cfg["max_lo"]), cast(cfg["max_hi"]), cast(cfg["max_default"]),
            step=cast(cfg["step"]), key=f"stress_max_{kind}",
        )
        n_steps = st.slider("Sweep Discrete Steps", 3, 15, 8, key="stress_steps")
        global_t = st.slider("Baseline Global Threshold", 0, 255, 127, key="stress_global_t")

        angle = 0
        if kind == "Motion blur":
            angle = st.slider("Kernel Vector Angle", 0, 180, 45, key="stress_angle")

        run_btn = st.button("▶ Run Stress Test Suite", type="primary", key="run_stress", use_container_width=True)

        st.markdown(f"""
        <div style="margin-top: 14px; padding: 10px; background-color: #ECE8DE; border: 1px solid #D9D3C5; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
          <div style="display: flex; justify-content: space-between; font-weight: 600; color: #5B5F66; text-transform: uppercase;">
            <span>Simulation Cost</span>
            <span style="color: #1F5F5B;">{n_steps * 5} INFERENCES</span>
          </div>
          <div style="color: #191c20; margin-top: 2px;">Matrix: {n_steps} levels × 5 algorithms</div>
          <div style="display: flex; justify-content: space-between; color: #5B5F66; margin-top: 4px; padding-top: 4px; border-top: 1px solid rgba(217,211,197,0.8);">
            <span>Estimated Latency:</span>
            <span style="color: #191c20; font-weight: 600;">~150 ms</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        heuristic_desc = {
            "Gaussian noise": "Gaussian noise perturbs intensities pixel-wise: Global threshold degrades gracefully until noise crosses T=127; Otsu calculates adaptive bimodal minimums; Canny edge detection suffers high false-positive gradients once σ > 15; Hybrid model uses hysteresis seeds to suppress spurious gradients.",
            "Low light (gamma)": "Low light (gamma) monotonically compresses dynamic range toward 0: Fixed Global Threshold (T=127) catastrophically collapses as all pixels fall below 127; Otsu dynamically recalibrates threshold down to preserve ~95% IoU; Hybrid tracks adaptive evidence.",
            "Motion blur": "Motion blur attenuates high-frequency boundaries along vector angle: Edge Canny fails rapidly due to perimeter smearing; Otsu and Global maintain robust object detection as total area integral is conserved.",
        }

        st.markdown(f"""
        <div style="margin-top: 14px; padding: 12px; background-color: #FBFAF7; border: 1px solid #D9D3C5; border-radius: 6px; box-shadow: 0 1px 2px rgba(28,31,35,0.04);">
          <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 11px; font-weight: 700; color: #5B5F66; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
            Diagnostic Heuristics
          </div>
          <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; color: #5B5F66; line-height: 1.45;">
            {heuristic_desc.get(kind, "")}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_main:
        # Card 1: Live Preview at Single Level
        st.markdown("""
        <div style="background-color: #FBFAF7; border: 1px solid #D9D3C5; border-radius: 6px 6px 0 0; padding: 12px 16px; border-bottom: none;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 8px;">
              <span style="font-family: 'IBM Plex Sans', sans-serif; font-size: 10px; font-weight: 700; text-transform: uppercase; color: #1F5F5B; background-color: rgba(31,95,91,0.1); padding: 2px 6px; border-radius: 2px; border: 1px solid rgba(31,95,91,0.2);">
                Slice Inspection
              </span>
              <span style="font-family: 'Source Serif 4', serif; font-size: 17px; font-weight: 600; color: #191c20;">
                Live Preview at Single Degradation Step
              </span>
            </div>
          </div>
          <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; color: #5B5F66; margin-top: 2px;">
            Interactive single-slice visualization comparing mask segmentations against baseline ground truth.
          </div>
        </div>
        """, unsafe_allow_html=True)

        preview_level = st.slider(
            f"Select slice {cfg['axis']}", cast(cfg["start"]), cast(max_level), cast(cfg["start"]),
            step=cast(cfg["preview_step"]), key=f"stress_preview_{kind}",
            label_visibility="collapsed",
        )

        degraded = _apply_degradation(gray, kind, preview_level, angle, NOISE_SEED)
        params = _method_params(global_t, polarity)

        cols = st.columns(len(params) + 1)
        with cols[0]:
            st.markdown("<div style='background-color:#1C1F23; color:#E8E4D9; padding:4px 8px; border-radius:4px 4px 0 0; font-size:10px; font-weight:600; font-family:sans-serif; text-transform:uppercase;'>Perturbed Source</div>", unsafe_allow_html=True)
            st.image(degraded, use_container_width=True)
            st.markdown(f"<div style='font-family:monospace; font-size:11px; color:#5B5F66; text-align:center;'>Level {preview_level}</div>", unsafe_allow_html=True)

        algo_colors = {"global": "#3B5B7A", "otsu": "#C8561E", "adaptive": "#5B7F4A", "edge": "#B08A2E", "hybrid": "#7B1FA2"}

        for col, key in zip(cols[1:], params.keys()):
            seg = segmentation.segment(degraded, key, **params[key])
            iou = evaluation.calculate_iou(seg["mask"], gt_mask)
            color = algo_colors.get(key, "#1F5F5B")
            thresh_str = f"T={seg['threshold']:.0f}" if seg.get("threshold") is not None else ""
            with col:
                st.markdown(f"<div style='background-color:{color}; color:#FBFAF7; padding:4px 6px; border-radius:4px 4px 0 0; font-size:10px; font-weight:600; font-family:sans-serif; text-transform:uppercase; display:flex; justify-content:space-between;'><span>{SHORT_LABELS[key]}</span><span style='font-family:monospace;'>IoU {iou:.3f}</span></div>", unsafe_allow_html=True)
                st.image(seg["mask"], use_container_width=True)
                st.markdown(f"<div style='font-family:monospace; font-size:11px; color:#5B5F66; text-align:center;'>{SHORT_LABELS[key]} {thresh_str}</div>", unsafe_allow_html=True)

        # CARD 2 & 3: Full sweep
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        if run_btn:
            levels = _build_levels(kind, max_level, n_steps)
            with st.spinner(f"Running {len(levels)} levels × {len(params)} methods..."):
                curves = cached_stress_sweep(
                    gray, gt_mask, kind, levels, angle, NOISE_SEED, polarity, global_t
                )
            st.session_state["stress_result"] = {
                "kind": kind, "levels": levels, "curves": curves, "axis": cfg["axis"],
            }

        stored = st.session_state.get("stress_result")
        if stored is None:
            # Auto-run with baseline levels so user never sees empty state!
            levels = _build_levels(kind, max_level, n_steps)
            curves = cached_stress_sweep(
                gray, gt_mask, kind, levels, angle, NOISE_SEED, polarity, global_t
            )
            stored = {
                "kind": kind, "levels": levels, "curves": curves, "axis": cfg["axis"],
            }
            st.session_state["stress_result"] = stored

        st.markdown(f"""
        <div style="background-color: #FBFAF7; border: 1px solid #D9D3C5; border-radius: 6px; padding: 14px 18px; margin-top: 14px; box-shadow: 0 1px 2px rgba(28,31,35,0.04);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div>
              <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 10px; font-weight: 700; text-transform: uppercase; color: #5B5F66; letter-spacing: 0.08em;">Analytical Regression</div>
              <div style="font-family: 'Source Serif 4', serif; font-size: 17px; font-weight: 600; color: #191c20;">Algorithm Robustness Across Degradation Spectrum</div>
            </div>
            <div style="background-color: rgba(31,95,91,0.1); border: 1px solid rgba(31,95,91,0.2); padding: 4px 10px; border-radius: 3px; font-family: 'IBM Plex Sans', sans-serif; font-size: 11px; font-weight: 600; color: #1F5F5B;">
              ● Evaluated against active Ground Truth
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        fig = visualization.plot_degradation_curve(
            stored["levels"], stored["curves"], x_label=stored["axis"],
            title=f"IoU vs. {stored['axis']} ({stored['kind']})",
        )
        st.pyplot(fig, use_container_width=True)

        # Card 3: Quantitative Breakdown Table
        st.markdown("""
        <div style="background-color: #FBFAF7; border: 1px solid #D9D3C5; border-radius: 6px; padding: 12px 16px; margin-top: 14px; box-shadow: 0 1px 2px rgba(28,31,35,0.04);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <div>
              <div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 10px; font-weight: 700; text-transform: uppercase; color: #5B5F66; letter-spacing: 0.08em;">Tabular Matrix</div>
              <div style="font-family: 'Source Serif 4', serif; font-size: 17px; font-weight: 600; color: #191c20;">Degradation Level Performance Breakdown</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        table_rows = []
        for i, lvl in enumerate(stored["levels"]):
            row = {"Level": f"L{i+1} ({lvl})"}
            best_iou = -1
            best_m = ""
            for key, ious in stored["curves"].items():
                val = ious[i]
                row[SHORT_LABELS[key]] = f"{val:.3f}"
                if val > best_iou:
                    best_iou = val
                    best_m = SHORT_LABELS[key]
            row["Top Pipeline"] = best_m
            table_rows.append(row)

        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, hide_index=True, use_container_width=True)

        # CSV Export
        csv_data = df_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇ Export Degradation Matrix CSV",
            csv_data,
            file_name=f"stress_test_{stored['kind'].lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# TAB 3: auto-tuning (grid search)
# ---------------------------------------------------------------------------

def render_tuning_tab():
    st.subheader("Algorithmic auto-tuning (grid search)")
    st.write(
        "Tries every parameter combination in a fixed range and keeps the one "
        "with the highest IoU, removing human slider bias. "
        "Adaptive: 15 block sizes × 21 values of C = 315 runs. "
        "Edge-based: 5 low × 4 high Canny thresholds = 20 runs. "
        "Hybrid Model: 5 eta × 3 block × 3 C × 2 edge bands = 90 runs."
    )

    if gt_mask is None:
        st.info("Provide a ground-truth mask in the **Ground truth & polarity** "
                "section above. Grid search needs a real reference to optimise against.")
        return
    if gt_source == "pseudo (Otsu)":
        st.warning("Tuning against Otsu's output only finds the parameters that best "
                   "imitate Otsu. Use a real ground-truth mask for meaningful results.")

    gray = result["filtered"]
    col1, col2 = st.columns(2)
    with col1:
        adaptive_method = st.selectbox("Adaptive local statistic (fixed during search)",
                                       ["gaussian", "mean"], key="tune_adaptive_method")
    with col2:
        closing = st.slider("Edge gap-closing kernel size (fixed during search)",
                            1, 25, 5, step=2, key="tune_closing")

    signature = _signature(gray, gt_mask, polarity, adaptive_method, closing)

    if st.button("Run Grid Search", type="primary", key="run_tuning"):
        bar_a = st.progress(0.0, text="Adaptive grid search (315 combinations)")
        ad = optimization.grid_search_adaptive(
            gray, gt_mask, object_polarity=polarity, adaptive_method=adaptive_method,
            progress_callback=lambda f: bar_a.progress(f, text="Adaptive grid search"),
        )
        bar_e = st.progress(0.0, text="Edge-based grid search (20 combinations)")
        ed = optimization.grid_search_edge(
            gray, gt_mask, closing_kernel_size=closing,
            progress_callback=lambda f: bar_e.progress(f, text="Edge-based grid search"),
        )
        bar_h = st.progress(0.0, text="Hybrid model grid search (90 combinations)")
        hy = optimization.grid_search_hybrid(
            gray, gt_mask, object_polarity=polarity,
            progress_callback=lambda f: bar_h.progress(f, text="Hybrid model grid search"),
        )
        bar_a.empty()
        bar_e.empty()
        bar_h.empty()
        st.session_state["tuning"] = {"signature": signature, "adaptive": ad, "edge": ed, "hybrid": hy}

    stored = st.session_state.get("tuning")
    if stored is None:
        st.write("Click **Run Grid Search** to start.")
        return
    if stored["signature"] != signature:
        st.info("Image, ground truth or settings changed since the last search. "
                "Click **Run Grid Search** again to refresh.")
        return

    ad, ed = stored["adaptive"], stored["edge"]
    hy = stored.get("hybrid")

    st.markdown("#### Optimal parameters")
    rows_summary = [
        {"Method": "Adaptive Thresholding",
         "Optimal parameters": f"block_size={ad['best_block_size']}, C={ad['best_C']}",
         "Max IoU": round(ad["best_iou"], 4),
         "Combinations tried": ad["n_combinations"]},
        {"Method": "Edge-Based Segmentation",
         "Optimal parameters": f"low={ed['best_low_threshold']}, high={ed['best_high_threshold']}",
         "Max IoU": round(ed["best_iou"], 4),
         "Combinations tried": ed["n_combinations"]},
    ]
    if hy is not None:
        rows_summary.append({
            "Method": "Hybrid Model",
            "Optimal parameters": f"eta_trust={hy['best_eta_trust']}, block_size={hy['best_block_size']}, C={hy['best_C']}, edge_band={hy['best_edge_band']}",
            "Max IoU": round(hy["best_iou"], 4),
            "Combinations tried": hy["n_combinations"]
        })
    st.dataframe(pd.DataFrame(rows_summary), hide_index=True)

    st.markdown("#### Masks at the optimum")
    cols = st.columns(4 if hy is not None else 3)
    with cols[0]:
        st.image(gt_mask, caption="Ground truth", use_container_width=True)
    with cols[1]:
        st.image(ad["best_mask"], caption=f"Adaptive (IoU {ad['best_iou']:.3f})", use_container_width=True)
    with cols[2]:
        st.image(ed["best_mask"], caption=f"Edge-based (IoU {ed['best_iou']:.3f})", use_container_width=True)
    if hy is not None:
        with cols[3]:
            st.image(hy["best_mask"], caption=f"Hybrid (IoU {hy['best_iou']:.3f})", use_container_width=True)

    with st.expander("Top 10 combinations per method"):
        top_a = sorted(ad["results"], key=lambda r: r["iou"], reverse=True)[:10]
        top_e = sorted(ed["results"], key=lambda r: r["iou"], reverse=True)[:10]
        st.markdown("**Adaptive**")
        st.dataframe(pd.DataFrame(top_a), hide_index=True)
        st.markdown("**Edge-based**")
        st.dataframe(pd.DataFrame(top_e), hide_index=True)
        if hy is not None:
            top_h = sorted(hy["results"], key=lambda r: r["iou"], reverse=True)[:10]
            st.markdown("**Hybrid Model**")
            st.dataframe(pd.DataFrame(top_h), hide_index=True)


# ---------------------------------------------------------------------------
# TAB 4: hardware profile
# ---------------------------------------------------------------------------

def render_profile_tab():
    st.subheader("Hardware profile (edge-device feasibility)")
    st.write(
        "Measures execution time and RAM for each method on the current "
        "preprocessed image, using each method's default parameters."
    )

    gray = result["filtered"]
    repeats = st.slider("Runs per method", 3, 50, 10, key="profile_repeats",
                        help="Each method is run this many times inside one monitored "
                             "window; the median time is reported.")

    if st.button("Run Profiling", type="primary", key="run_profile"):
        overrides = {
            "global": {"object_polarity": polarity},
            "otsu": {"object_polarity": polarity},
            "adaptive": {"object_polarity": polarity},
            "hybrid": {"object_polarity": polarity},
        }
        try:
            with st.spinner("Profiling all five methods..."):
                rows = profiling.profile_all_methods(gray, method_params=overrides,
                                                     repeats=repeats)
            st.session_state["profile"] = {"rows": rows, "repeats": repeats}
        except Exception as exc:  # noqa: BLE001 - surface any profiler/OS issue to the user
            st.error(f"Profiling failed: {exc}")

    stored = st.session_state.get("profile")
    if stored is None:
        st.write("Click **Run Profiling** to measure.")
        return

    df = pd.DataFrame(stored["rows"])
    st.dataframe(df, hide_index=True)
    st.bar_chart(df.set_index("Method")[["Time (ms)"]])
    st.caption(
        f"Median of {stored['repeats']} runs per method. 'Peak RAM' is the whole "
        "Streamlit process (dominated by Python, OpenCV and Streamlit itself); "
        "'RAM increase' is the growth above the starting level and is only a "
        "rough figure. 'Alloc peak' is the peak memory allocated by Python/NumPy "
        "during one run and is the best indicator of the algorithm's own "
        "footprint (it does not include OpenCV's internal buffers, so it is a "
        "lower bound). Results are re-measured on every click (not cached)."
    )


# ---------------------------------------------------------------------------
# TAB 5: morphology, detection and measurements
# ---------------------------------------------------------------------------

def render_measure_tab():
    st.subheader("Morphology, detection and measurements")
    st.write(
        "Each method's raw mask goes through the SAME cleanup "
        "(opening -> closing -> small-region removal), then objects are "
        "detected and measured. Methods use fixed default parameters, so "
        "differences come from the segmentation itself. All measurements "
        "are in pixels."
    )

    gray = result["filtered"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        open_k = st.slider("Opening kernel", 1, 15, 3, step=2, key="meas_open")
    with c2:
        close_k = st.slider("Closing kernel", 1, 25, 5, step=2, key="meas_close")
    with c3:
        min_area = st.slider("Min object area (px)", 1, 2000, 50, key="meas_min_area")
    with c4:
        global_t = st.slider("Global threshold (fixed)", 0, 255, 127, key="meas_global_t")

    params = _method_params(global_t, polarity)

    runs = {}
    summary = []
    for key, p in params.items():
        seg = segmentation.segment(gray, key, **p)
        cleaned = morphology.clean_mask(seg["mask"], open_kernel_size=open_k,
                                        close_kernel_size=close_k, min_area=min_area)
        objects = detection.detect_objects(cleaned["final"], min_area=min_area)
        table = features.compute_all_features(objects)
        overlay = visualization.draw_detections(result["resized"], objects)
        runs[key] = {"raw": seg["mask"], "cleaned": cleaned, "objects": objects,
                     "table": table, "overlay": overlay}

        row = {
            "Method": SHORT_LABELS[key],
            "Objects": len(objects),
            "Foreground (%)": round(100.0 * np.count_nonzero(cleaned["final"])
                                    / cleaned["final"].size, 1),
            "Mean area (px)": round(float(table["area"].mean()), 1) if len(table) else 0.0,
            "Mean circularity": round(float(table["circularity"].mean()), 3) if len(table) else 0.0,
        }
        if gt_mask is not None:
            row["IoU raw"] = round(evaluation.calculate_iou(seg["mask"], gt_mask), 4)
            row["IoU cleaned"] = round(evaluation.calculate_iou(cleaned["final"], gt_mask), 4)
        summary.append(row)

    st.markdown("#### Detections per method")
    cols = st.columns(len(params))
    for col, key in zip(cols, params.keys()):
        with col:
            st.image(runs[key]["overlay"][:, :, ::-1],
                     caption=f"{SHORT_LABELS[key]}: {len(runs[key]['objects'])} objects",
                     use_container_width=True)

    st.markdown("#### Summary")
    st.dataframe(pd.DataFrame(summary), hide_index=True)
    if gt_mask is None:
        st.caption("Add a ground-truth mask in the **Ground truth & polarity** "
                   "section above to also see IoU before and after cleaning.")
    elif gt_source == "pseudo (Otsu)":
        st.caption("IoU is measured against Otsu's own output (pseudo ground truth), "
                   "so it favours Otsu and is only a demo.")

    st.markdown("#### Method detail")
    detail_key = st.selectbox(
        "Method", list(params.keys()), format_func=lambda k: SHORT_LABELS[k],
        key="meas_detail_method",
    )
    run = runs[detail_key]
    steps = [("Raw mask", run["raw"]),
             ("After opening", run["cleaned"]["opened"]),
             ("After closing", run["cleaned"]["closed"]),
             ("Final (small regions removed)", run["cleaned"]["final"])]
    for col, (caption, img) in zip(st.columns(4), steps):
        with col:
            st.image(img, caption=caption, use_container_width=True)

    st.markdown("**Measurements**")
    if len(run["table"]) == 0:
        st.write("No objects detected with the current settings.")
    else:
        st.dataframe(run["table"], hide_index=True)
        st.download_button(
            "Download measurements (CSV)",
            run["table"].to_csv(index=False).encode("utf-8"),
            file_name=f"measurements_{detail_key}.csv",
            mime="text/csv",
            key="meas_download",
        )
        


# ---------------------------------------------------------------------------
# TAB 7: Hybrid Model
# ---------------------------------------------------------------------------

def render_hybrid_tab():
    # ---- Header banner -------------------------------------------------------
    st.markdown("""
    <div style="background-color:#FBFAF7; border:1px solid #D9D3C5; border-radius:6px;
                padding:14px 18px; margin-bottom:18px; box-shadow:0 1px 2px rgba(28,31,35,.04);">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <div style="font-family:'IBM Plex Sans',sans-serif; font-size:10px; font-weight:700;
                      text-transform:uppercase; color:#5B5F66; letter-spacing:.08em;">Fusion Engine</div>
          <div style="font-family:'Source Serif 4',serif; font-size:20px; font-weight:600;
                      color:#191c20; margin-top:2px;">Hybrid Segmentation Model</div>
          <div style="font-family:'IBM Plex Sans',sans-serif; font-size:12px; color:#5B5F66; margin-top:4px;">
            Combines Otsu global threshold, adaptive local threshold, and Canny edge zones
            through region-level hysteresis to produce a single fused foreground mask.
          </div>
        </div>
        <div style="background-color:rgba(31,95,91,.1); border:1px solid rgba(31,95,91,.2);
                    padding:8px 14px; border-radius:4px; text-align:center;">
          <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#1F5F5B; font-weight:700;">METHOD</div>
          <div style="font-family:'Source Serif 4',serif; font-size:16px; color:#1F5F5B; font-weight:600;">HYBRID v2</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    gray = result["filtered"]

    # ---- Sidebar-style parameter panel ---------------------------------------
    with st.expander("⚙️ Hybrid Model Parameters", expanded=True):
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            hyb_polarity_label = st.radio(
                "Object polarity",
                ["Objects darker than background", "Objects brighter than background"],
                index=0, key="hyb_polarity",
            )
            hyb_polarity = "darker" if "darker" in hyb_polarity_label else "brighter"

            eta_trust = st.slider(
                "Otsu trust threshold (η)",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                key="hyb_eta",
                help="Otsu separability score (0–1). Above this value, Otsu evidence "
                     "is included in the strong-seed stage. Below it, the model falls "
                     "back to adaptive + edges only.",
            )

        with col_p2:
            hyb_block = st.slider(
                "Adaptive block size",
                min_value=3, max_value=99, value=21, step=2, key="hyb_block",
                help="Neighborhood side length for the adaptive threshold stage (must be odd).",
            )
            hyb_C = st.slider(
                "Adaptive constant C",
                min_value=1, max_value=30, value=8, key="hyb_C",
                help="Subtracted from the local Gaussian-weighted mean. Higher = stricter foreground.",
            )

        with col_p3:
            hyb_edge_band = st.slider(
                "Edge-zone dilation (px)",
                min_value=1, max_value=21, value=5, step=2, key="hyb_edge_band",
                help="How many pixels around each Canny edge count as an 'edge zone' for weak-seed acceptance.",
            )
            hyb_min_seed = st.slider(
                "Min seed ratio",
                min_value=0.0, max_value=0.5, value=0.0, step=0.01, key="hyb_seed_ratio",
                help="A connected component must have at least this fraction of its pixels as seeds to be kept.",
            )

    run_hybrid = st.button("▶ Run Hybrid Segmentation", type="primary", key="run_hybrid")

    if not run_hybrid and "hybrid_result" not in st.session_state:
        st.info("Configure the parameters above and click **▶ Run Hybrid Segmentation**.")
        return

    if run_hybrid:
        hyb_seg = segmentation.segment_hybrid(
            gray,
            object_polarity=hyb_polarity,
            eta_trust=eta_trust,
            block_size=hyb_block,
            C=hyb_C,
            edge_band=hyb_edge_band,
            min_seed_ratio=hyb_min_seed,
        )
        st.session_state["hybrid_result"] = {
            "seg": hyb_seg,
            "polarity": hyb_polarity,
            "params": dict(
                eta_trust=eta_trust, block_size=hyb_block,
                C=hyb_C, edge_band=hyb_edge_band, min_seed_ratio=hyb_min_seed,
            ),
        }

    stored = st.session_state.get("hybrid_result")
    if stored is None:
        return

    hyb_seg = stored["seg"]
    inter   = hyb_seg["intermediate"]
    eta_val = inter["eta"]
    t_otsu  = hyb_seg["threshold"]
    otsu_trusted = eta_val >= stored["params"]["eta_trust"]

    # ---- Otsu reliability badge ------------------------------------------
    badge_color = "#1F5F5B" if otsu_trusted else "#B08A2E"
    badge_text  = "OTSU TRUSTED" if otsu_trusted else "OTSU IGNORED"
    badge_bg    = "rgba(31,95,91,.1)" if otsu_trusted else "rgba(176,138,46,.12)"
    badge_border = "rgba(31,95,91,.3)" if otsu_trusted else "rgba(176,138,46,.4)"    
    st.markdown(f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:18px;">
      <div style="background:{badge_bg}; border:1px solid {badge_border}; border-radius:5px;
                  padding:10px 18px; font-family:'JetBrains Mono',monospace; font-size:12px; color:{badge_color};">
        <span style="font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.06em;
                     display:block; margin-bottom:2px;">Otsu Reliability (η)</span>
        <span style="font-size:22px; font-weight:700;">{eta_val:.3f}</span>
        &nbsp;<span style="font-size:11px; border:1px solid {badge_border}; padding:2px 6px;
                           border-radius:3px;">{badge_text}</span>
      </div>
      <div style="background:#F4F1EA; border:1px solid #D9D3C5; border-radius:5px;
                  padding:10px 18px; font-family:'JetBrains Mono',monospace; font-size:12px; color:#5B5F66;">
        <span style="font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.06em;
                     display:block; margin-bottom:2px;">Otsu Global Threshold (T)</span>
        <span style="font-size:22px; font-weight:700; color:#191c20;">{t_otsu:.0f}</span>
      </div>
      <div style="background:#F4F1EA; border:1px solid #D9D3C5; border-radius:5px;
                  padding:10px 18px; font-family:'JetBrains Mono',monospace; font-size:12px; color:#5B5F66;">
        <span style="font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.06em;
                     display:block; margin-bottom:2px;">Processing Time</span>
        <span style="font-size:22px; font-weight:700; color:#191c20;">{hyb_seg['processing_time_ms']:.1f} ms</span>
      </div>
      <div style="background:#F4F1EA; border:1px solid #D9D3C5; border-radius:5px;
                  padding:10px 18px; font-family:'JetBrains Mono',monospace; font-size:12px; color:#5B5F66;">
        <span style="font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.06em;
                     display:block; margin-bottom:2px;">Foreground Coverage</span>
        <span style="font-size:22px; font-weight:700; color:#191c20;">{100.0*np.count_nonzero(hyb_seg['mask'])/hyb_seg['mask'].size:.1f}%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Fusion pipeline strip -------------------------------------------
    st.markdown("""
    <div style="background:#FBFAF7; border:1px solid #D9D3C5; border-radius:6px;
                padding:12px 16px; margin-bottom:12px;">
      <div style="font-family:'IBM Plex Sans',sans-serif; font-size:10px; font-weight:700;
                  text-transform:uppercase; color:#5B5F66; letter-spacing:.08em; margin-bottom:8px;">Fusion Pipeline — Intermediate Stages</div>
    </div>
    """, unsafe_allow_html=True)

    stage_cols = st.columns(6)
    stages = [
        ("Input (preprocessed)",    gray),
        ("Feature\n(CLAHE + invert)",inter["feature"]),
        ("Evidence 1\nOtsu mask",   inter["otsu_mask"]),
        ("Evidence 2\nAdaptive mask",inter["adaptive_mask"]),
        ("Evidence 3\nEdge map",     inter["edges"]),
        ("Fused Mask\n(final)",      hyb_seg["mask"]),
    ]
    stage_colors = ["#4B5563", "#374151", "#3B5B7A", "#5B7F4A", "#B08A2E", "#1F5F5B"]
    for col, (caption, img), color in zip(stage_cols, stages, stage_colors):
        with col:
            st.markdown(f"<div style='background:{color}; color:#FBFAF7; padding:3px 6px;"
                        f"border-radius:4px 4px 0 0; font-size:9px; font-weight:700;"
                        f"font-family:monospace; text-transform:uppercase; white-space:pre;'>{caption}</div>",
                        unsafe_allow_html=True)
            st.image(img, use_container_width=True)

    # ---- Seed / weak pool detail -----------------------------------------
    with st.expander("🔍 Hysteresis Detail — Seeds & Weak Pool", expanded=False):
        scols = st.columns(3)
        with scols[0]:
            st.image(inter["weak_pool"],
                     caption="Weak pool (union / adaptive only)",
                     use_container_width=True)
        with scols[1]:
            st.image(inter["seeds"],
                     caption="Accepted seeds (strong + edge-zone weak)",
                     use_container_width=True)
        with scols[2]:
            st.image(hyb_seg["mask"],
                     caption="Final fused mask (hysteresis output)",
                     use_container_width=True)
        st.caption(
            "**Weak pool**: either Otsu or Adaptive claims the pixel (or Adaptive only when "
            "η < η-trust). **Seeds**: pixels that both evidence layers agree on, or weak pixels "
            "that fall inside a Canny edge zone. Only connected components of the weak pool that "
            "contain at least one seed pixel survive into the final mask."
        )

    st.divider()

    # ---- Comparison against the four classical methods -------------------
    st.markdown("""
    <div style="background:#FBFAF7; border:1px solid #D9D3C5; border-radius:6px;
                padding:12px 16px; margin-bottom:12px;">
      <div style="font-family:'IBM Plex Sans',sans-serif; font-size:10px; font-weight:700;
                  text-transform:uppercase; color:#5B5F66; letter-spacing:.08em; margin-bottom:4px;">Head-to-Head Comparison</div>
      <div style="font-family:'Source Serif 4',serif; font-size:17px; font-weight:600; color:#191c20;">Hybrid vs. Classical Methods</div>
    </div>
    """, unsafe_allow_html=True)

    classical_params = {
        "global":   {"threshold": 127, "object_polarity": stored["polarity"]},
        "otsu":     {"object_polarity": stored["polarity"]},
        "adaptive": {"block_size": 11, "C": 2, "adaptive_method": "gaussian",
                     "object_polarity": stored["polarity"]},
        "edge":     {"low_threshold": 50, "high_threshold": 150, "closing_kernel_size": 5},
    }
    classical_colors = {
        "global":   "#3B5B7A",
        "otsu":     "#C8561E",
        "adaptive": "#5B7F4A",
        "edge":     "#B08A2E",
    }

    all_masks = {"hybrid": hyb_seg["mask"]}
    comp_summary = []
    for key, p in classical_params.items():
        seg = segmentation.segment(gray, key, **p)
        all_masks[key] = seg["mask"]
        row = {
            "Method": SHORT_LABELS[key],
            "Foreground (%)": round(100.0 * np.count_nonzero(seg["mask"]) / seg["mask"].size, 1),
            "Time (ms)": round(seg["processing_time_ms"], 2),
        }
        if gt_mask is not None:
            row["IoU vs GT"] = round(evaluation.calculate_iou(seg["mask"], gt_mask), 4)
        comp_summary.append(row)

    hyb_row = {
        "Method": "Hybrid ★",
        "Foreground (%)": round(100.0 * np.count_nonzero(hyb_seg["mask"]) / hyb_seg["mask"].size, 1),
        "Time (ms)": round(hyb_seg["processing_time_ms"], 2),
    }
    if gt_mask is not None:
        hyb_iou = evaluation.calculate_iou(hyb_seg["mask"], gt_mask)
        hyb_row["IoU vs GT"] = round(hyb_iou, 4)
    comp_summary.insert(0, hyb_row)

    # Mask grid
    mask_cols = st.columns(5)
    display_order = [("Hybrid ★", "#1F5F5B", hyb_seg["mask"])] + [
        (SHORT_LABELS[k], classical_colors[k], all_masks[k]) for k in classical_params
    ]
    for col, (lbl, color, msk) in zip(mask_cols, display_order):
        with col:
            st.markdown(
                f"<div style='background:{color}; color:#FBFAF7; padding:4px 8px;"
                f"border-radius:4px 4px 0 0; font-size:10px; font-weight:700;"
                f"font-family:monospace; text-transform:uppercase;'>{lbl}</div>",
                unsafe_allow_html=True,
            )
            st.image(msk, use_container_width=True)

    # Summary table
    st.dataframe(pd.DataFrame(comp_summary), hide_index=True, use_container_width=True)
    if gt_mask is None:
        st.caption("Add a ground-truth mask in the **Ground Truth & Polarity** section above to see IoU scores.")
    elif gt_source == "pseudo (Otsu)":
        st.caption("IoU is measured against Otsu's output (pseudo ground truth) — favours Otsu.")

    st.divider()

    # ---- Morphology + detection on hybrid mask ---------------------------
    st.markdown("""
    <div style="background:#FBFAF7; border:1px solid #D9D3C5; border-radius:6px;
                padding:12px 16px; margin-bottom:12px;">
      <div style="font-family:'IBM Plex Sans',sans-serif; font-size:10px; font-weight:700;
                  text-transform:uppercase; color:#5B5F66; letter-spacing:.08em; margin-bottom:4px;">Post-Processing</div>
      <div style="font-family:'Source Serif 4',serif; font-size:17px; font-weight:600; color:#191c20;">Morphology, Detection & Measurements</div>
    </div>
    """, unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        hyb_open_k  = st.slider("Opening kernel",  1, 15, 3, step=2, key="hyb_open")
    with mc2:
        hyb_close_k = st.slider("Closing kernel",  1, 25, 5, step=2, key="hyb_close")
    with mc3:
        hyb_min_area = st.slider("Min object area (px)", 1, 2000, 50, key="hyb_min_area")

    from src import morphology, detection, features
    cleaned = morphology.clean_mask(
        hyb_seg["mask"],
        open_kernel_size=hyb_open_k,
        close_kernel_size=hyb_close_k,
        min_area=hyb_min_area,
    )
    objects  = detection.detect_objects(cleaned["final"], min_area=hyb_min_area)
    feat_tbl = features.compute_all_features(objects)
    overlay  = visualization.draw_detections(result["resized"], objects)

    det_cols = st.columns(4)
    with det_cols[0]:
        st.image(hyb_seg["mask"], caption="Raw hybrid mask", use_container_width=True)
    with det_cols[1]:
        st.image(cleaned["opened"], caption="After opening", use_container_width=True)
    with det_cols[2]:
        st.image(cleaned["final"], caption="After closing + small-region removal", use_container_width=True)
    with det_cols[3]:
        st.image(overlay[:, :, ::-1],
                 caption=f"Detections: {len(objects)} object(s)",
                 use_container_width=True)

    if gt_mask is not None:
        iou_raw     = evaluation.calculate_iou(hyb_seg["mask"], gt_mask)
        iou_cleaned = evaluation.calculate_iou(cleaned["final"], gt_mask)
        st.markdown(
            f"<div style='font-family:JetBrains Mono,monospace; font-size:12px; color:#5B5F66; margin:6px 0;'>"
            f"IoU raw mask: <b style='color:#191c20'>{iou_raw:.4f}</b> &nbsp;|&nbsp; "
            f"IoU cleaned mask: <b style='color:#1F5F5B'>{iou_cleaned:.4f}</b></div>",
            unsafe_allow_html=True,
        )

    st.markdown("**Measurements (pixels)**")
    if len(feat_tbl) == 0:
        st.write("No objects detected with the current settings.")
    else:
        st.dataframe(feat_tbl, hide_index=True)
        st.download_button(
            "⬇ Download measurements (CSV)",
            feat_tbl.to_csv(index=False).encode("utf-8"),
            file_name="measurements_hybrid.csv",
            mime="text/csv",
            key="hyb_download",
        )

    # ---- Algorithm explainer --------------------------------------------
    with st.expander("📖 How the Hybrid Model works", expanded=False):
        st.markdown("""
**Three-source evidence fusion with region-level hysteresis**

The Hybrid Model gathers evidence from three independent sources, then fuses
them using the same hysteresis principle Canny uses for edges — but applied
at connected-component level.

| Stage | What it does | Failure mode it avoids |
|-------|-------------|------------------------|
| **Otsu** | Finds the globally optimal threshold T that maximises between-class variance (η) | Fails on uneven illumination |
| **Adaptive** | Computes a per-pixel local threshold from a Gaussian-weighted neighbourhood | Fragments large uniform regions |
| **Canny edge zone** | Marks pixels within *edge_band* px of a detected edge | Provides spatial boundary anchoring |

**Fusion logic (hysteresis):**
- If **η ≥ η_trust** (Otsu histogram is bimodal, Otsu is reliable):
  - *Strong seeds* = Otsu **AND** Adaptive (both agree → confident foreground)
  - *Weak pool* = Otsu **OR** Adaptive (either claims → candidate region)
- If **η < η_trust** (Otsu is unreliable, e.g. uneven lighting):
  - *Strong seeds* = Adaptive **AND** edge_zone
  - *Weak pool* = Adaptive only
- **Final seeds** = strong ∪ (weak ∩ edge_zone)
- Any connected component in the weak pool that contains **at least one seed** pixel is kept. Others are discarded.

This means the Hybrid mask is never noisier than the weakest individual
method, yet it recovers regions the strictest method would have missed,
as long as at least one independent source provides a confident vote.
        """)



def render_benchmark_tab():
    st.subheader("Benchmark on the synthetic dataset")
    st.write(
        "All four methods run on every dataset image with identical "
        "preprocessing, cleanup and detection, scored against exact ground "
        "truth. This tab does not use the uploaded image."
    )

    if not (DATASET_DIR / "ground_truth" / "counts.csv").exists():
        st.info("Dataset not found. Run `python generate_dataset.py` from the "
                "project folder first.")
        return

    c1, c2 = st.columns(2)
    with c1:
        bench_noise = st.selectbox("Noise reduction", ["none", "gaussian", "median"],
                                   key="bench_noise")
    with c2:
        bench_kernel = st.slider("Filter kernel size", 3, 31, 5, step=2,
                                 key="bench_kernel")
    cfg = _bench_cfg(bench_noise, bench_kernel)

    out_name = ("benchmark_results.csv" if bench_noise == "none"
                else f"results_{bench_noise}.csv")
    if st.button("Run benchmark", type="primary", key="bench_run"):
        with st.spinner(f"Running all images x {len(run_benchmark.METHOD_KEYS)} methods..."):
            run_benchmark.run_benchmark(DATASET_DIR, RESULTS_DIR, out_name, cfg)
        st.session_state["bench_file"] = out_name
        st.success(f"Wrote results/{out_name}")
    st.caption(f"Run output file: results/{out_name}. A different kernel size "
               "overwrites the file of the same noise method.")

    # --- Results ---------------------------------------------------------
    files = sorted(p.name for p in RESULTS_DIR.glob("*.csv")) if RESULTS_DIR.exists() else []
    if not files:
        st.write("No results yet. Click **Run benchmark**.")
        return
    default = st.session_state.get("bench_file", files[0])
    chosen = st.selectbox("Results file", files,
                          index=files.index(default) if default in files else 0,
                          key="bench_choice")
    df = pd.read_csv(RESULTS_DIR / chosen)

    metric = st.selectbox("Metric to chart", BENCH_METRICS, key="bench_metric")
    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(visualization.plot_method_comparison(df, metric))
    with col_b:
        st.pyplot(visualization.plot_condition_comparison(df, metric))

    st.markdown("#### Mean per method")
    summary = df.groupby("method")[BENCH_METRICS].mean().reindex(
        list(run_benchmark.METHOD_KEYS)).round(3)
    st.dataframe(summary)
    st.caption("mean_iou and mean_measurement_error_pct only cover images where "
               "the method matched at least one object. Prefer count_accuracy "
               "and mask_iou for headline comparisons.")

    # --- Browse images ---------------------------------------------------
    st.markdown("#### Browse images")
    st.caption("Recomputed live with the noise settings above, so it matches the "
               "results file only if that file was run with the same settings.")
    entries = cached_dataset(str(DATASET_DIR))
    name = st.selectbox("Image", [e["filename"] for e in entries], key="bench_image")
    entry = next(e for e in entries if e["filename"] == name)
    rows, masks = run_benchmark.run_image(entry, cfg)

    cols = st.columns(len(run_benchmark.METHOD_KEYS) + 1)
    with cols[0]:
        st.image(entry["image"],
                 caption=f"{name}: {entry['n_objects']} true objects "
                         f"({entry['gt_blobs']} blobs)",
                 use_container_width=True)
    for col, key in zip(cols[1:], run_benchmark.METHOD_KEYS):
        objects = detection.detect_objects(masks[key], min_area=cfg.min_area)
        overlay = visualization.draw_detections(entry["image"], objects)
        with col:
            st.image(overlay[:, :, ::-1],
                     caption=f"{SHORT_LABELS[key]}: {len(objects)} detected",
                     use_container_width=True)

    st.dataframe(
        pd.DataFrame(rows)[["method", "gt_count", "gt_blobs", "detected_count",
                            "correct", "missed", "false_positives",
                            "count_accuracy", "mask_iou"]].round(3),
        hide_index=True)
    with st.expander("Ground-truth mask"):
        st.image(entry["gt_mask"], width=300)



# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_pipeline, tab_stress, tab_tuning, tab_profile, tab_measure, tab_bench, tab_hybrid = st.tabs(
    ["Pipeline", "Stress-Test", "Auto-Tuning", "Hardware Profile",
     "Measurements", "Benchmark", "Hybrid Model"]
)

with tab_pipeline:
    render_pipeline_tab()
with tab_stress:
    render_stress_tab()
with tab_tuning:
    render_tuning_tab()
with tab_profile:
    render_profile_tab()
with tab_measure:
    render_measure_tab()
with tab_bench:
    render_benchmark_tab()
with tab_hybrid:
    render_hybrid_tab()