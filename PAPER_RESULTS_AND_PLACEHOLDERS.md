# Paper Results, Tables, and Placeholder Guide

This document contains all measured quantitative results, hardware profiling benchmarks, dataset parameters, hypothesis statuses, and exact text replacements needed to complete the manuscript:
**"Comparative Image Segmentation and Object Measurement System with a Novel Hybrid Multi-Evidence Fusion Model"**.

---

## 1. Paper Metadata & Author Information (Pages 1 & 2)

| Section / Field | Original Yellow Placeholder | Suggested Value |
| :--- | :--- | :--- |
| **Top Left Header** | `[VENUE WEBSITE]` | URL of journal/conference or leave empty if preprint |
| **Top Left Header** | `[PEER-REVIEW STATUS — NOT YET SUBMITTED]` | `Under Review` or `Preprint` |
| **Top Right Header** | `[TARGET JOURNAL / CONFERENCE NAME]` | `IEEE Transactions on Image Processing` or target conference name |
| **Top Right Header** | `ISSN: [TO BE ASSIGNED BY VENUE]` | Remove placeholder or leave for publisher |
| **Top Right Header** | `Vol. [X] Issue [X], [MONTH YEAR]` | `Vol. 1, Issue 1, March 2026` |
| **Authors (p. 1)** | `Authors; Vedant Patil [CONFIRM FINAL AUTHOR LIST AND ORDER]` | `Vedant Patil` (add co-authors/advisors if applicable) |
| **Affiliation (p. 1)** | `[CONFIRM INSTITUTION AND OFFICIAL AFFILIATION]` | `Department of Computer Engineering, [Your Institute / University Name], [City, Country]` |
| **Project Repository** | `Project Repository: [INSERT VERIFIED REPOSITORY URL]` | `https://github.com/snaksartrate/ip_project_2` |
| **Footer (Left)** | `[MANUSCRIPT ID]` | Omit or enter venue manuscript ID |
| **Footer (Center)** | `[LICENSE STATEMENT — CONFIRM WITH TARGET VENUE]` | `Creative Commons Attribution 4.0 International (CC BY 4.0)` |

---

## 2. Experimental Setup Specifications (Pages 7 & 8)

### Page 7 — Protocol & Reproducibility Settings
* **Preprocessing defaults (Sec. IV-D):**
  * Maximum image dimension: `800 px` (aspect ratio preserved).
  * Grayscale conversion: $Y = 0.299R + 0.587G + 0.114B$.
  * Baseline denoising: `None` (optional Gaussian / Median filter with $k = 5$).
* **Stress-test sweep settings (Sec. IV-D):**
  * Evaluated across `8 discrete levels` per degradation channel on `normal_01.png` ($640 \times 480\text{ px}$), directional motion blur vector angle = $45^\circ$.
* **Aggregation statistic (Sec. IV-F):**
  * `mean ± standard deviation across n = 5 images per condition (N = 20 total)`.
* **Reproducibility tokens (Sec. IV-F):**
  * Random seed: `42`
  * Git commit hash: `d703440187fa8a86fa7f3d70e8b82af1d4ee6dd9`
  * Software versions: Python 3.14.5 (or 3.11+), OpenCV 5.0.0, NumPy 2.4.2, pandas 3.0.0, Streamlit 1.63.0, Matplotlib 3.11.2.

---

### Page 8 — Table III: Software and Hardware Configuration

| Component | Specification |
| :--- | :--- |
| **Language** | Python 3.11+ (tested on Python 3.14.5) |
| **Image processing** | OpenCV (`cv2`) 5.0.0 (or 4.x) |
| **Numerical / data** | NumPy 2.4.2; pandas 3.0.0 |
| **Interface / plotting** | Streamlit 1.63.0; Matplotlib 3.11.2 |
| **Memory profiling** | `tracemalloc` (Python standard library), `psutil` |
| **CPU / cores** | **Apple M4 (10 cores: 4 performance + 6 efficiency)** |
| **RAM** | **16.0 GB Unified Memory** |
| **GPU** | **Apple Silicon integrated GPU (Not used / CPU compute only)** |
| **Operating system** | **macOS Sequoia (macOS 15/26 arm64)** |

---

### Page 8 — Table IV: Synthetic Dataset Conditions

| Condition | Description | Segmentation challenge | Generator settings |
| :--- | :--- | :--- | :--- |
| **normal** | Clean grayscale background; circular/elliptical foreground objects | Baseline (expected high performance for all methods) | **5 images, 640×480 px, 5–8 objects each (radius 22–48 px, 30% ellipses), mild sensor noise ($\sigma = 3$)** |
| **noisy** | Gaussian noise added to the clean image | Noise robustness | **Gaussian noise $\sigma = 12$ plus salt-and-pepper noise (3% of pixels corrupted)** |
| **uneven** | Gradient or vignette background illumination | Violates global-threshold assumption | **Linear brightness ramp from 1.0 down to ~0.35–0.50 at random angle, plus sensor noise $\sigma = 3$** |
| **touching** | Objects overlapping or adjacent | Object separation and counting | **2–3 clusters per image (2–4 objects per cluster), radial overlap 0–3 px, plus sensor noise $\sigma = 3$** |

---

## 3. Hyperparameters (Page 10, Table V)

| Module | Parameter | Value | Note |
| :--- | :--- | :--- | :--- |
| **Shared post-proc.** | `opening / closing kernel` | $3 \times 3$ / $5 \times 5$ (elliptical) | Same for all methods |
| **Shared post-proc.** | `min_area` | $50\text{ px}$ | Object filter |
| **Evaluation** | `match IoU threshold` | $0.5$ | Greedy one-to-one |
| **Hybrid** | `CLAHE` | clip 2.0, $8 \times 8$ tiles | Before thresholding |
| **Hybrid** | `pre-blur` | $5 \times 5$ Gaussian | Working image B |
| **Hybrid** | `object_polarity` | darker | Image inverted if darker |
| **Hybrid** | `eta_trust` ($\eta_{\text{trust}}$) | $0.50$ | Fusion-branch gate |
| **Hybrid** | `block_size / C` | $21$ / $8$ | Adaptive evidence |
| **Hybrid** | `edge_band` | $5\text{ px}$ | Edge-zone dilation |
| **Hybrid** | `min_seed_ratio` ($\rho$) | $0.0$ | Only $\ge 1$ seed required |
| **Hybrid** | `min_area_frac` ($f_{\text{min}}$) | $10^{-5}$ | With floor of $10\text{ px}$ |
| **Hybrid** | `Canny low / high` | $0.5 T^* / T^*$ | Derived from Otsu |
| **Stress: noise** | `variance max` | $1600$ ($\sigma \le 40$) | Seed 42 |
| **Stress: gamma** | `gamma max` ($\gamma$) | $4.0$ | From $1.0$ |
| **Stress: blur** | `length max` | $31\text{ px}$ | From $1\text{ px}$ (angle $45^\circ$) |
| **Adaptive grid** | `b, C` | $15 \times 21 = 315$ | Section III-L |
| **Edge grid** | `low, high` | $5 \times 4 = 20$ | Closing kernel fixed: **5 px (elliptical)** |
| **Preprocessing** | `denoise kernel, max_dimension` | **k = 5, max_dimension = 800 px** | Standard shared pipeline settings |
| **Global** | `threshold T` | **127** | Midpoint intensity baseline |
| **Adaptive (classical)** | `default b, C` | **b = 11, C = 2** | Gaussian weighting |

---

## 4. Quantitative Results & Evaluation Tables

### Page 11 — Table VII: Overall Comparison Over Dataset (20 Images)

*All numbers are empirical means across all 20 images.*

| Method | Mask IoU ↑ | Object IoU ↑ | Count acc. ↑ | Meas. err. (%) ↓ | FP ↓ | Missed ↓ | Time (ms) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Global threshold** | **0.7778** | **0.9411** | **0.8102** | **9.02%** | **0.55** | **1.80** | **0.02 ms** |
| **Otsu** | **0.7794** | **0.9419** | **0.8102** | **9.03%** | **0.55** | **1.80** | **0.08 ms** |
| **Adaptive** | **0.2830** | **0.9517** | **0.6286** | **68.02%** | **3.95** | **1.35** | **0.42 ms** |
| **Edge-based** | **0.7444** | **0.9080** | **0.6712** | **14.09%** | **0.55** | **2.65** | **0.57 ms** |
| **Hybrid (proposed)** | **0.7825** | **0.9336** | **0.8702** | **9.84%** | **0.60** | **1.50** | **2.90 ms** |

---

### Page 12 — Table VIII: Mean Mask IoU and Count Accuracy by Condition

| Method | Normal IoU | Normal Cnt | Noisy IoU | Noisy Cnt | Uneven IoU | Uneven Cnt | Touching IoU | Touching Cnt |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Global threshold** | **0.9960** | **1.0000** | **0.9920** | **1.0000** | **0.1304** | **0.7267** | **0.9929** | **0.5143** |
| **Otsu** | **0.9994** | **1.0000** | **0.9926** | **1.0000** | **0.1286** | **0.7267** | **0.9969** | **0.5143** |
| **Adaptive** | **0.2542** | **1.0000** | **0.3647** | **0.0000** | **0.2656** | **1.0000** | **0.2475** | **0.5143** |
| **Edge-based** | **0.9688** | **1.0000** | **0.0753** | **0.1705** | **0.9649** | **1.0000** | **0.9685** | **0.5143** |
| **Hybrid (proposed)**| **0.9963** | **1.0000** | **0.9892** | **1.0000** | **0.1503** | **0.9667** | **0.9943** | **0.5143** |

---

### Page 13 — Table IX: Hardware Profiling Benchmark
*Measured over $N = 10$ repeats on `normal_01.png` ($640 \times 480\text{ px}$) on Apple M4:*

| Method | Median time (ms) | Peak RAM (MB) | RAM increase (MB) | Alloc peak (MB) |
| :--- | :---: | :---: | :---: | :---: |
| **Global threshold** | **0.141** | **173.516** | **0.000** | **2.932** |
| **Otsu** | **0.205** | **173.531** | **0.000** | **2.932** |
| **Adaptive** | **0.531** | **173.578** | **0.047** | **2.932** |
| **Edge-based** | **0.501** | **173.703** | **0.156** | **3.518** |
| **Hybrid (proposed)**| **2.898** | **177.625** | **3.953** | **9.087** |

---

### Page 14 — Table X: Separability Score and Hybrid Fusion Branch Analysis

| Quantity | Normal | Noisy | Uneven | Touching |
| :--- | :---: | :---: | :---: | :---: |
| **Mean $\eta$ ($\pm$ s.d.)** | **$0.9654 \pm 0.0077$** | **$0.8541 \pm 0.0167$** | **$0.6273 \pm 0.0399$** | **$0.9665 \pm 0.0042$** |
| **Fraction with $\eta \ge \eta_{\text{trust}}$ ($0.50$)** | **1.00 (100%)** | **1.00 (100%)** | **1.00 (100%)** | **1.00 (100%)** |
| **Hybrid IoU $-$ best classical IoU** | **$-0.0031$** | **$-0.0039$** | **$-0.8146$** | **$-0.0026$** |

> **Critical Analytical Finding (Section V-F):** On uneven illumination images, CLAHE's local contrast enhancement raises $\eta$ to a mean of $0.6273$, which exceeds $\eta_{\text{trust}} = 0.50$. As a result, the intended fallback branch ($\eta < \eta_{\text{trust}}$) is **never activated** on uneven illumination images. Because the high-$\eta$ branch relies on Otsu, the Hybrid model inherits Otsu's failure mode on uneven images ($0.1503$ IoU vs. Edge's $0.9649$ IoU).

---

## 5. Hypotheses Verification (Page 11, Table VI)

| ID | Expected Behaviour | Test | Status & Measured Findings |
| :--- | :--- | :--- | :--- |
| **H1** | Normal condition: all methods comparable | Table VIII, normal | **[CONFIRMED]** Otsu ($0.9994$), Hybrid ($0.9963$), Global ($0.9960$), and Edge ($0.9688$) all achieve near-perfect masks. |
| **H2** | Noise: edge-based degrades fastest | Fig. 9(a) | **[CONFIRMED]** Edge mask IoU collapses to $0.0753$ under noise (performance cliff at variance $400$). |
| **H3** | Uneven illumination: global & Otsu fail; edge reasonable | Table VIII, uneven | **[CONFIRMED for Baselines / PARTIAL for Hybrid]** Global ($0.1304$) and Otsu ($0.1286$) collapse; Edge succeeds ($0.9649$). Hybrid achieves $0.1503$ due to CLAHE inflating $\eta$. |
| **H4** | Touching objects: count accuracy drops to ~0.51 while IoU stays high | Table VIII, touching | **[CONFIRMED]** Every single method achieves exactly **$0.5143$** count accuracy due to connected-component merging, while mask IoU remains $> 0.96$. |
| **H5** | Low light: global fails at moderate $\gamma$; Otsu recalibrates | Fig. 9(b) | **[CONFIRMED]** Global collapses at $\gamma = 2.5$ ($0.085$ IoU), while Otsu recalibrates downward, maintaining IoU $\ge 0.968$ up to $\gamma = 4.0$. |
| **H6** | Motion blur: edge degrades fastest; thresholding robust | Fig. 9(c) | **[CONFIRMED]** Edge collapses at blur length $11\text{ px}$ (IoU $\to 0.000$), while Global, Otsu, and Hybrid retain IoU $> 0.92$ up to $31\text{ px}$. |
| **H7** | Hybrid: low-$\eta$ branch activation | Table X | **[REFUTED]** Low-$\eta$ branch was inactive across all uneven images because CLAHE elevated $\eta$ above $0.50$. |
| **H8** | Hybrid runtime remains competitive | Table IX | **[CONFIRMED]** Hybrid median latency is $2.898\text{ ms}$ on CPU, well within real-time interactive requirements ($< 5\text{ ms}$). |

---

## 6. Stress Test Performance Cliffs (Page 8, Section V-D)

Evaluated on `normal_01.png` with seed 42:

1. **Gaussian Noise Sweep (Variance $0$ to $1600$):**
   * **Edge-based:** Suffers an abrupt breakdown at **Variance 400 ($\sigma = 20$)**, where IoU drops from $0.970$ to $0.085$. Spurious gradients cause flood-fill leakage across the entire frame.
   * **Global & Otsu:** Remain exceptionally robust ($> 0.965$ IoU even at Variance $1600$).
   * **Hybrid:** Retains high overlap ($0.913$ IoU at Variance $1600$).

2. **Low-Light Gamma Sweep ($\gamma = 1.0$ to $4.0$):**
   * **Global threshold ($T=127$):** Suffers a performance cliff at **$\gamma = 2.5$**, where IoU drops from $0.935$ to $0.085$ because both objects and background fall below the static threshold.
   * **Otsu & Hybrid:** Dynamically re-estimate the threshold valley, preserving IoU $\ge 0.968$ up to $\gamma = 4.0$.

3. **Motion Blur Sweep (Length $1$ to $31\text{ px}$, $45^\circ$):**
   * **Edge-based:** Experiences a complete performance cliff at **Length 11 px**, falling to $0.000$ IoU as boundary attenuation destroys edge connectivity.
   * **Global, Otsu, & Hybrid:** Remain stable with IoU $> 0.92$ up to $31\text{ px}$.

---

## 7. What Should Come in the FIGURE PLACEHOLDERS

| Figure | Location | Content & How to Generate |
| :--- | :--- | :--- |
| **Figure 6** | Page 9 | **Synthetic Dataset Conditions:** 2 rows × 4 columns grid showing the 4 test images (`normal_01.png`, `noisy_01.png`, `uneven_01.png`, `touching_01.png`) on the top row, and their corresponding exact ground-truth binary masks on the bottom row. Direct exports from `dataset/images/` and `dataset/ground_truth/`. |
| **Figure 7** | Page 9 | **Comparative Performance Bar Chart:** Grouped bar chart comparing the 5 algorithms on Mask IoU, Count Accuracy, and Measurement Error (%) generated from Table VII. |
| **Figure 8** | Page 12 | **Hybrid Model Stage Breakdown:** Sequential multi-panel image showing: (1) Grayscale input, (2) CLAHE-enhanced image, (3) Otsu mask, (4) Gaussian adaptive mask, (5) Canny edge zone, (6) Region hysteresis seeds, and (7) Output masks of all 5 methods. Exported from Tab 7 ("Hybrid Model") in the Streamlit app. |
| **Figure 9** | Page 13 | **Stress-Test Robustness Curves:** Three subplots plotting IoU vs. Severity for: (a) Gaussian noise variance ($0–1600$), (b) Low-light gamma ($1.0–4.0$), and (c) Motion-blur length ($1–31\text{ px}$, $45^\circ$). Curves for all 5 methods. Exported from Tab 2 ("Stress-Test") in the app. |
| **Figure 10** | Page 14 | **Hardware & Memory Profile:** Two-panel bar plot: Left panel shows median execution time (ms) on Apple M4 ($0.14\text{ ms}$ to $2.90\text{ ms}$); Right panel shows process peak RAM ($~173–177\text{ MB}$) vs. algorithm dynamic allocation peak ($2.9\text{ MB}$ to $9.1\text{ MB}$). Exported from Tab 4 ("Hardware Profile"). |
| **Figure 11** | Page 14 | **Measurements & Feature Extraction Overlay:** Screenshot from Tab 5 ("Measurements"): Image overlay showing bounding boxes, centroids, and contour IDs, alongside the geometric feature table (Area, Perimeter, Circularity, Equivalent Diameter). |

---

## 8. Exact Replacement Text for Narrative Placeholders

### Abstract (Page 1):
> **Replace `[INSERT MEASURED SUMMARY AFTER BENCHMARK RUNS]` with:**  
> "The benchmark evaluation across 20 synthetic images demonstrates that the proposed Hybrid model achieves the highest overall count accuracy (87.0%) and top-tier mask IoU (0.7825), outperforming classical global (0.7778), Otsu (0.7794), edge-based (0.7444), and default adaptive (0.2830) methods. Stress testing reveals that edge-based segmentation experiences a severe performance cliff under Gaussian noise (variance $\ge 400$) and motion blur ($\ge 11\text{ px}$), whereas Otsu and the Hybrid model maintain IoU $> 0.91$ up to variance $1600$ and low-light $\gamma = 4.0$. Hardware profiling on an Apple M4 CPU confirms median latencies under $3.0\text{ ms}$ for all methods, validating edge feasibility."

### Section I-D / Contributions (Page 2):
> **Replace `[ARE PENDING]` with:**  
> "are confirmed in Section V: the hybrid model achieves the highest overall count accuracy (87.0%) and top-tier mask IoU (0.7825), while executing in 2.90 ms."

### Section V-C / Condition Analysis (Page 8):
> **Replace `[INSERT AFTER BENCHMARK]` with:**  
> "On normal images, Otsu (0.9994), Hybrid (0.9963), and Global (0.9960) achieve near-perfect segmentation. Under noise, edge-based segmentation collapses to 0.0753 IoU. Under uneven illumination, global and Otsu fail severely (~0.13 IoU) while edge-based reaches 0.9649. For touching objects, all methods exhibit exactly 0.5143 count accuracy due to connected-component merging, while preserving high mask IoU (>0.96)."

### Section V-D / Performance Cliffs (Page 8):
> **Replace `[INSERT PER METHOD AND CHANNEL]` with:**  
> "The performance cliffs occur at: Gaussian noise variance 400 ($\sigma = 20$) for edge-based segmentation (IoU drops to 0.085); low-light gamma $\gamma = 2.5$ for global thresholding (IoU drops to 0.085); and directional motion blur length 11 px for edge-based segmentation (IoU drops to 0.000)."

### Section VI-A / Discussion (Page 9):
> **Replace `[WILL BE ADDED AFTER MEASUREMENT]` with:**  
> "The empirical findings demonstrate clear trade-offs: no single classical algorithm succeeds under all four imaging regimes. The hybrid model successfully bridges noise and clean illumination regimes, but analysis reveals that CLAHE preprocessing inflates Otsu's separability score on uneven lighting ($\eta = 0.627 \ge 0.50$), preventing the edge-adaptive fallback branch from triggering."

### Section VIII / Conclusion (Page 11):
> **Replace `( [CONCLUDE FROM MEASURED RESULTS] )` with:**  
> "The measured results demonstrate that the Hybrid Multi-Evidence Fusion Model achieves superior object detection accuracy (87.0%) and robust mask overlap across diverse stress tests, while identifying CLAHE-induced separability bias under uneven illumination as a primary direction for future adaptive gating refinement."
