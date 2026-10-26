import os
import base64
import mimetypes

OUTPUT_FILE = "imagebase.txt"
FILE_SEPARATOR = "=" * 100
CONTENT_SEPARATOR = "-" * 100

# Only these images are included in the output.
DATASET_IMAGES = [
    "dataset/images/normal_01.png",
    "dataset/images/noisy_01.png",
    "dataset/images/uneven_01.png",
    "dataset/images/touching_01.png",
]
GROUND_TRUTH_IMAGES = [
    "dataset/ground_truth/normal_01.png",
    "dataset/ground_truth/noisy_01.png",
    "dataset/ground_truth/uneven_01.png",
    "dataset/ground_truth/touching_01.png",
    "dataset/ground_truth/instances/touching_01.png",
]
METHODS = ["global", "otsu", "adaptive", "edge"]
SAMPLES = ["normal_01", "noisy_01", "uneven_01", "touching_01"]

# Baseline masks created by:
# python run_benchmark.py --save-masks --out-file figures_run.csv
BASELINE_MASKS = [
    f"results/masks/{method}/{sample}.png"
    for sample in SAMPLES
    for method in METHODS
]

# Median-filter masks created with the separate output folder requested in the
# instructions. The script checks both common layouts without scanning broadly.
MEDIAN_MASK_CANDIDATES = [
    f"results_fig_median/masks/{method}/{sample}.png"
    for sample in SAMPLES
    for method in METHODS
] + [
    f"results_fig_median/{method}/{sample}.png"
    for sample in SAMPLES
    for method in METHODS
]

# Figure 5 adaptive masks, if produced by a separate script.
ADAPTIVE_FAILURE_CANDIDATES = [
    "results/adaptive_normal_01_block11_C2.png",
    "results/adaptive_normal_01_block11_C15.png",
    "results/adaptive_normal_01_block99_C8.png",
    "results_fig_adaptive/normal_01_block11_C2.png",
    "results_fig_adaptive/normal_01_block11_C15.png",
    "results_fig_adaptive/normal_01_block99_C8.png",
]

REQUESTED_PATHS = DATASET_IMAGES + GROUND_TRUTH_IMAGES + BASELINE_MASKS + MEDIAN_MASK_CANDIDATES + ADAPTIVE_FAILURE_CANDIDATES


def dump_imagebase(root_dir: str) -> None:
    """Write only the explicitly requested research-figure images to imagebase.txt."""
    root_dir = os.path.abspath(root_dir)
    output_path = os.path.abspath(os.path.join(root_dir, OUTPUT_FILE))
    script_path = os.path.abspath(__file__)

    with open(output_path, "w", encoding="utf-8") as out_file:
        written = set()

        for rel_path in REQUESTED_PATHS:
            file_path = os.path.abspath(os.path.join(root_dir, rel_path))

            if file_path in {output_path, script_path} or file_path in written:
                continue

            if not os.path.isfile(file_path):
                continue

            try:
                with open(file_path, "rb") as image_file:
                    encoded = base64.b64encode(image_file.read()).decode("ascii")
            except OSError:
                continue

            mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"

            out_file.write(FILE_SEPARATOR + "\\n")
            out_file.write(f"{rel_path} begins here\\n")
            out_file.write(CONTENT_SEPARATOR + "\\n")
            out_file.write(f"mime_type: {mime_type}\\n")
            out_file.write("encoding: base64\\n\\n")

            # Wrap Base64 across multiple lines for readability, like codebase.py.
            for index in range(0, len(encoded), 100):
                out_file.write(encoded[index:index + 100] + "\\n")

            out_file.write(CONTENT_SEPARATOR + "\\n")
            out_file.write(f"eof {rel_path}\\n")
            out_file.write(FILE_SEPARATOR + "\\n\\n")

            written.add(file_path)

    print(f"Image dump written to {output_path}")

if __name__ == "__main__":
    dump_imagebase(os.getcwd())
