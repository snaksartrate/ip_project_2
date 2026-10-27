# Comparative Image Segmentation and Object Measurement System
## Detailed Guide to Application Pages, Tabs, and UI Components

This document provides a complete breakdown of all the pages, tabs, interactive controls, data visualizations, and underlying processing workflows in the **Comparative Image Segmentation and Object Measurement System**.

---

## 1. Application Architecture Overview

The system is built with **Streamlit** (`app.py`), structured as a multi-stage, multi-tab modular dashboard. The user interface is organized into:

1. **Global Sidebar**: Preprocessing parameters and experiment toggles.
2. **Top Section (Input & Preprocessing)**: Image upload, validation, and shared ground-truth configuration.
3. **6 Functional Application Pages (Tabs)**:
   - **Tab 1: Pipeline** (Interactive Step-by-Step Image Processing & Segmentation)
   - **Tab 2: Stress-Test** (Robustness & Degradation Sweeps)
   - **Tab 3: Auto-Tuning** (Algorithmic Grid Search Optimization)
   - **Tab 4: Hardware Profile** (Feasibility & Resource Benchmarking)
   - **Tab 5: Measurements** (Morphology, Object Detection & Geometric Features)
   - **Tab 6: Benchmark** (Batch Dataset Evaluation, Comparison & Image Browser)

---

## 2. Global Sections (Shared Across All Tabs)

Before navigating into individual tabs, the application defines persistent controls and inputs in the global scope:

### 2.1 Sidebar: Preprocessing Settings (`⚙️ Settings`)
Located in the left collapsible sidebar expander:
* **Max image dimension (`slider`, 200–2000 px, step 50, default 800 px)**: Downscales images larger than this limit while preserving aspect ratio.
* **Noise reduction method (`radio`: `None`, `Gaussian`, `Median`, default `None`)**:
  * *Gaussian*: Bell-curve smoothing for sensor noise.
  * *Median*: Rank filter for impulse (salt-and-pepper) noise, preserving edges.
* **Filter kernel size (`slider`, 1/3–31, step 2, default 5)**: Defines neighborhood window size (must be an odd integer).
* **Show optional histogram equalization experiment (`checkbox`, default unchecked)**: Toggles an experimental contrast adjustment view without mutating the main pipeline.

### 2.2 Section 1: Image Upload & Validation
* **File Uploader**: Accepts `JPG`, `JPEG`, `PNG`, and `BMP` files.
* **Validation & Extraction**:
  * Validates file readability and format via `src/preprocessing.py`.
  * Extracts metadata: Height, Width, Channels, Dtype, and Size (KB).
  * Runs initial preprocessing pipeline: Downscaling $\rightarrow$ Grayscale conversion ($0.299R + 0.587G + 0.114B$) $\rightarrow$ Noise reduction filtering $\rightarrow$ Intensity histogram computation.

### 2.3 Shared Ground-Truth & Polarity Expander
Shared across Stress-Test, Auto-Tuning, Hardware Profile, and Measurements tabs:
* **Object polarity selector (`radio`)**:
  * `Objects darker than background`
  * `Objects brighter than background`
* **Ground-truth mask uploader (`file_uploader`)**:
  * Binary mask upload (`PNG`, `JPG`, `JPEG`, `BMP`) where white ($255$) = object, black ($0$) = background.
  * Automatically matches aspect ratio and resizes to the processed image dimensions.
* **Pseudo Ground-Truth Toggle (`checkbox`)**:
  * *"No mask? Use Otsu's output as a pseudo ground truth (demo only)"*.
  * Enables test exploration when no annotated reference mask is available.
* **Ground-Truth Indicator**: Shows the active mask source and the foreground coverage percentage along with a thumbnail preview.

---

## 3. Tab-by-Tab Page Breakdown

---

### Page / Tab 1: Pipeline
**File / Handler:** `render_pipeline_tab()` in `app.py`  
**Purpose:** Interactive, step-by-step visual demonstration of the end-to-end classical image processing and segmentation pipeline on the uploaded image.

#### On-Page Content & Sections:
1. **Section 2: Original Image & Metadata**
   - Displays the unmodified image.
   - Property table showing: Height (px), Width (px), Channels, Data Type, and Approximate file size (KB).
2. **Section 3: Resizing**
   - Details whether resizing occurred based on the maximum dimension constraint.
   - Shows before/after dimensions and displays the resized image.
3. **Section 4: Grayscale Conversion**
   - Details the luminance weighting equation ($Y = 0.299R + 0.587G + 0.114B$).
   - Side-by-side comparison: Before (color) vs. After (grayscale).
4. **Section 5: Noise Reduction**
   - Explains the active filter mechanism (None, Gaussian blur, or Median filter).
   - Side-by-side comparison: Grayscale input vs. Filtered output.
5. **Section 6: Intensity Histogram**
   - Matplotlib histogram plot of pixel intensities ($0–255$) on the filtered image.
   - *(Optional)* Histogram Equalization experiment displaying equalized image and equalized histogram side-by-side.
6. **Section 7: Segmentation Module**
   - **Method Selector (`selectbox`)**:
     1. `Global Thresholding`
     2. `Otsu Thresholding`
     3. `Adaptive Thresholding`
     4. `Edge-Based Segmentation`
   - **Method-Specific Controls**:
     - *Global Thresholding*: Threshold value slider ($0–255$, default $127$) + Polarity.
     - *Otsu Thresholding*: Automatic bimodal threshold selection (no slider needed) + Polarity.
     - *Adaptive Thresholding*: Block size slider ($3–51$, odd step, default $11$), Constant $C$ slider ($-20$ to $+20$, default $2$), Local statistic (`gaussian` or `mean`) + Polarity.
     - *Edge-Based Segmentation*: Canny low threshold slider ($0–255$, default $50$), Canny high threshold slider ($0–255$, default $150$), Gap-closing morphological kernel slider ($1–25$, step 2, default $5$).
   - **Action Button**: `"Run Segmentation"`
   - **Outputs Generated**:
     - Input grayscale image.
     - Binary output mask ($255$ = foreground, $0$ = background).
     - Method Information Table: Execution time (ms), Threshold value applied, and exact parameters used.
     - *For Edge-Based method*: 4-stage visual breakdown columns: Grayscale $\rightarrow$ Canny edge map $\rightarrow$ Gap-closed edges $\rightarrow$ Flood-filled binary mask.

---

### Page / Tab 2: Stress-Test
**File / Handler:** `render_stress_tab()` in `app.py`  
**Purpose:** Evaluates and plots the robustness and resilience of all four segmentation algorithms under synthetic real-world image degradations at increasing severity.

#### On-Page Content & Sections:
1. **Requirements Notice**: Requires a ground-truth mask or pseudo-GT from the shared inputs.
2. **Degradation Configuration**:
   - **Degradation Type Selector (`selectbox`)**:
     - *Gaussian Noise*: Tests resilience against additive zero-mean sensor noise.
     - *Low Light (Gamma)*: Tests non-linear darkening ($I_{out} = 255 \cdot (I_{in}/255)^\gamma$).
     - *Motion Blur*: Tests directional camera/object movement blur via a line PSF kernel.
   - **Severity Sliders**:
     - Maximum severity slider (Noise variance: $100–4000$; Gamma: $1.5–6.0$; Blur length: $3–61$ px).
     - Number of test levels ($3–15$, default $8$).
     - Fixed Global threshold slider ($0–255$, default $127$).
     - Motion angle slider ($0^\circ–180^\circ$, active only for Motion Blur).
3. **Single-Level Live Preview**:
   - Real-time slider to test a single degradation severity level immediately.
   - 5-column visualization: Degraded input image alongside the 4 resulting segmentation masks (Global, Otsu, Adaptive, Edge-based) annotated with their live IoU scores against the ground truth.
4. **Full Degradation Sweep**:
   - **Action Button**: `"Run Stress Test"` (uses cached function `cached_stress_sweep`).
   - **Degradation Curve Plot**: Matplotlib chart of IoU (y-axis) vs. Degradation Level (x-axis) across all 4 algorithms.
   - **Quantitative Data Table**: Full matrix listing degradation levels and exact IoU values for Global, Otsu, Adaptive, and Edge-based segmentation.

---

### Page / Tab 3: Auto-Tuning
**File / Handler:** `render_tuning_tab()` in `app.py`  
**Purpose:** Performs exhaustive algorithmic grid search over parameter spaces to find optimal segmentation parameters that maximize IoU against ground truth, eliminating human tuning bias and establishing oracle upper bounds.

#### On-Page Content & Sections:
1. **Grid Search Configuration**:
   - Adaptive local statistic selector (`gaussian` or `mean`).
   - Edge-based gap-closing kernel size slider ($1–25$, default $5$).
   - Displays search space scale:
     - Adaptive: $15 \text{ block sizes} \times 21 \text{ C values} = 315 \text{ runs}$.
     - Edge-based: $5 \text{ low thresholds} \times 4 \text{ high thresholds} = 20 \text{ runs}$.
2. **Execution & Progress**:
   - **Action Button**: `"Run Grid Search"`.
   - Dual animated progress bars tracking Adaptive and Edge-based search completion.
   - State caching with content hash signature to detect when settings or images change.
3. **Optimization Results**:
   - **Optimal Parameters Table**:
     - Method name.
     - Optimal parameter combinations (e.g., `block_size=99, C=8` or `low=30, high=120`).
     - Maximum achieved IoU score.
     - Total combinations evaluated.
   - **Visual Comparison at Optimum**:
     - 3-column view: Ground truth mask, Adaptive best mask (with IoU), and Edge-based best mask (with IoU).
   - **Top 10 Parameter Rankings Expander**:
     - Tables detailing the top 10 best-performing parameter combinations and their exact IoU scores for both methods.

---

### Page / Tab 4: Hardware Profile
**File / Handler:** `render_profile_tab()` in `app.py`  
**Purpose:** Profiles computational execution time and memory footprint (RAM and Python heap allocation) to evaluate edge-device feasibility and real-time processing capability.

#### On-Page Content & Sections:
1. **Profile Controls**:
   - Runs per method slider ($3–50$, default $10$): Runs repeated iterations inside an isolated monitoring window and calculates median time.
   - **Action Button**: `"Run Profiling"`.
2. **Benchmarked Metrics Table**:
   - Columns reported for each method:
     - `Method`: Algorithm label.
     - `Time (ms)`: Median runtime in milliseconds.
     - `Peak RAM (MB)`: Peak total process RSS memory via `psutil`.
     - `RAM increase (MB)`: Growth above baseline during execution.
     - `Alloc peak (KB)`: Exact heap memory allocated by Python/NumPy via `tracemalloc`.
3. **Execution Time Bar Chart**:
   - Visual bar chart comparing runtimes across Global, Otsu, Adaptive, and Edge-based methods.
4. **Feasibility Explanations**:
   - Explains the distinction between process RAM (dominated by Python runtime and OpenCV buffers) versus true algorithmic footprint (`tracemalloc`).

---

### Page / Tab 5: Measurements
**File / Handler:** `render_measure_tab()` in `app.py`  
**Purpose:** Applies a shared post-segmentation morphological cleanup and connected component detection pipeline across all methods, extracting per-object geometric measurements.

#### On-Page Content & Sections:
1. **Morphology & Detection Parameter Controls**:
   - Opening kernel size slider ($1–15$, default $3$): Removes isolated foreground noise.
   - Closing kernel size slider ($1–25$, default $5$): Fills internal holes within objects.
   - Minimum object area slider ($1–2000$ px, default $50$ px): Filters out spurious detections.
   - Global threshold slider ($0–255$, default $127$).
2. **Visual Detection Overlays**:
   - 4-column side-by-side comparison displaying bounding boxes, centroids, and labels overlaid on the resized color image for each method.
3. **Summary Comparison Table**:
   - Compares all 4 segmentation methods on:
     - `Objects`: Total detected object count.
     - `Foreground (%)`: Percentage of image classified as object.
     - `Mean area (px)`: Average detected object area.
     - `Mean circularity`: Average shape circularity ($4\pi \cdot \text{Area} / \text{Perimeter}^2$).
     - `IoU raw` & `IoU cleaned`: IoU before and after morphological cleanup (if GT is loaded).
4. **Method Detail & Pipeline Inspector**:
   - Dropdown to inspect a specific method in detail.
   - **4-Stage Mask Evolution Columns**:
     1. Raw binary mask directly from segmentation.
     2. Mask after morphological opening.
     3. Mask after morphological closing.
     4. Final mask after small-region removal.
5. **Per-Object Feature Table & Export**:
   - Full dataframe displaying per-object geometric features:
     - `id`: Object identifier.
     - `area`: Area in pixels.
     - `perimeter`: Contour boundary length.
     - `centroid_x`, `centroid_y`: Coordinate location.
     - `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`: Bounding box dimensions.
     - `equivalent_diameter`: Diameter of a circle with the same area.
     - `circularity`: Compactness ($1.0 = \text{perfect circle}$).
     - `aspect_ratio`: Bounding box width / height.
     - `solidity`: Ratio of contour area to convex hull area.
   - **Export Button**: `"Download measurements (CSV)"` for downstream offline analysis.

---

### Page / Tab 6: Benchmark
**File / Handler:** `render_benchmark_tab()` in `app.py`  
**Purpose:** Comprehensive batch benchmarking on a standardized 20-image synthetic dataset across 4 conditions (`normal`, `noisy`, `uneven`, `touching`), scored against exact ground-truth instance maps and masks.

#### On-Page Content & Sections:
1. **Batch Runner Configuration**:
   - Noise reduction dropdown (`none`, `gaussian`, `median`).
   - Filter kernel size slider ($3–31$, default $5$).
   - **Action Button**: `"Run benchmark"` (executes `run_benchmark.py` logic and writes CSV to `results/`).
2. **Results File Selector**:
   - Dropdown to select existing benchmark CSV runs (e.g., `benchmark_results.csv`, `results_median.csv`, `results_tuned.csv`).
3. **Comparative Analysis Charts**:
   - Metric selector dropdown supporting:
     - `count_accuracy` ($1.0 = \text{exact match}$)
     - `mask_iou` (post-cleanup pixel overlap)
     - `mean_iou` (IoU of matched objects)
     - `raw_mask_iou` (pre-cleanup pixel overlap)
     - `false_positives` (spurious detected objects)
     - `missed` (unmatched true objects)
     - `mean_measurement_error_pct` (geometric error on matched objects)
     - `processing_time_ms` (execution runtime)
   - **Side-by-side Matplotlib Charts**:
     - *Method Comparison Chart*: Aggregate bar plot across the four algorithms.
     - *Condition Comparison Chart*: Grouped chart showing performance across `normal`, `noisy`, `uneven`, and `touching` conditions.
4. **Mean Performance Summary Table**:
   - Aggregated metric averages grouped by method across the entire 20-image dataset.
5. **Interactive Dataset Image Browser**:
   - Dropdown to inspect any of the 20 dataset images individually.
   - 5-column display:
     - Ground-truth image annotated with true count and connected component blobs.
     - 4 detection overlay images with detection boxes for each method.
   - Detailed per-image metric table showing ground-truth counts, detected counts, correct matches, false positives, count accuracy, and mask IoU.
   - Collapsible expander displaying the exact ground-truth binary mask.

---

## 4. Summary Mapping of UI Tabs to Core Modules

| Application Page / Tab | Primary Source Modules (`src/`) | Supporting CLI Script |
|---|---|---|
| **Sidebar & Input** | `preprocessing.py` | — |
| **Pipeline (Tab 1)** | `preprocessing.py`, `segmentation.py`, `visualization.py` | — |
| **Stress-Test (Tab 2)** | `degradation.py`, `segmentation.py`, `evaluation.py`, `visualization.py` | — |
| **Auto-Tuning (Tab 3)** | `optimization.py`, `segmentation.py`, `evaluation.py` | `run_tuned_benchmark.py` |
| **Hardware Profile (Tab 4)** | `profiling.py`, `segmentation.py` | — |
| **Measurements (Tab 5)** | `morphology.py`, `detection.py`, `features.py`, `visualization.py` | — |
| **Benchmark (Tab 6)** | `evaluation.py`, `detection.py`, `morphology.py`, `visualization.py` | `run_benchmark.py`, `run_comparison.py`, `generate_dataset.py` |

---
