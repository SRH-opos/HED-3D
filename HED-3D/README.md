# HED-3D

HED-3D estimates one normalized relative-depth value for each engine-oil fill-port region of interest (ROI) from a single RGB image. The framework combines YOLO11m-based region proposals, hierarchical ellipse localization, and a MobileNetV3-Large student network with deterministic and evidential heads. The two depth estimates are fused into an ROI-level scalar prediction using SoftGate.

This repository provides the implementation, inference and training entry points, evaluation utilities, and selected aggregate results reported in the manuscript.

> **Repository layout:** The project is contained in the `HED-3D/` subdirectory of the current GitHub repository. After cloning the repository, change into that directory before running the commands below:
>
> ```bash
> cd HED-3D
> ```

## What is included

| Directory | Contents |
| --- | --- |
| `hed3d/` | ROI inference, pseudo-label preparation, student training, evaluation, and plotting utilities |
| `weights/` | The author-provided DualKD inference checkpoint and its provenance information |
| `paper/` | Aggregate values reported for Table 10 and Figure 11, including the data used to redraw the risk–coverage plot |
| `docs/` | Method equations, evaluation protocol, input formats, and the scope and limitations of the public package |
| `configs/` | YOLO training and dataset configuration templates |

The repository contains code and selected aggregate result files. It does **not** contain the commercial image dataset, its annotations, or the complete per-image records used for the manuscript's experiments.

## Data availability and access restrictions

The source code is publicly released to support methodological transparency and reuse of the HED-3D implementation. The images and corresponding annotations used in this study originate from a commercial application and are subject to confidentiality restrictions. Therefore, the original dataset—including the training, validation, and test images and their annotations—is not publicly available.

The manuscript's reported aggregate metrics and the corresponding plotting data included in `paper/` are provided for reference. They do not replace the original data and are not sufficient to independently recompute every metric from the 302-image test set. Reproducing the full experimental results requires authorized access to the original ROI images, labels, or the relevant per-image prediction records.

Please do not commit confidential images, annotations, vehicle identifiers, calibration records, or per-sample outputs to this public repository.

## Environment setup

Use Python 3.10 or newer. Create and activate a virtual environment, then install a PyTorch and torchvision build compatible with your device. See the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

```bash
python -m venv .venv
# Activate the virtual environment for your shell.
python -m pip install -r requirements.txt
```

Run the commands in this README from the `HED-3D/` project directory (the directory containing `requirements.txt` and the `hed3d/` package).

The public inference checkpoint is documented in [`weights/README.md`](weights/README.md). The frozen distillation teacher, fine-tuned YOLO detector, and external Depth Anything V2 ViT-L checkpoint are not bundled here; obtain any required third-party models from their respective official sources and comply with their licenses and terms.

## Run inference

### Pre-cropped ROI

For a pre-cropped ROI image, run the student model and SoftGate using the released checkpoint:

```bash
python -m hed3d.run path/to/your_roi.png \
  --roi-input \
  --student weights/dualkd_student.pth \
  --out runs/roi
```

Replace `path/to/your_roi.png` with a local ROI image that you are authorized to use.

### Full image

For full-image inference, provide your own fine-tuned oil-fill-port detector checkpoint. Use `--aligned` when ellipse-aligned crops are required.

```bash
python -m hed3d.run path/to/your_images \
  --yolo path/to/your_oil_port_detector.pt \
  --aligned \
  --student weights/dualkd_student.pth \
  --dav2-repo path/to/Depth-Anything-V2 \
  --dav2-weights path/to/depth_anything_v2_vitl.pth \
  --out runs/images
```

The Depth Anything V2 repository and checkpoint are external dependencies and must be obtained separately. The `--cloud` option screens teacher-derived thin-ring points with the student's scalar depth reference and exports a local relative point set with pixel coordinates and scaled relative depth. The student network itself outputs one scalar per ROI; the exported point set should not be interpreted as a dense depth map predicted by the student.

## Prepare pseudo-labels

The pseudo-label preparation workflow uses an external Depth Anything V2 repository and checkpoint. Supply local images that you are authorized to process, together with the required model files, using the full-image workflow above. See [`docs/reliability.md`](docs/reliability.md) for the equations, data conventions, and evaluation boundaries.

## Train the student model

Prepare local training and validation CSV manifests with the columns:

```text
image,Z_rel_cam,vehicle_id
```

Keep the vehicle-level test split separate from the training and validation data. Rows generated by the pseudo-label preparation tool may also contain confidence metadata. The training objective reads the scalar target and uses the loss weights described in [`docs/reliability.md`](docs/reliability.md).

A separate pretrained dual-head checkpoint can provide frozen distillation targets:

```bash
python -m hed3d.train_student \
  --root path/to/your_roi_images \
  --train-csv path/to/train.csv \
  --val-csv path/to/validation.csv \
  --kd-checkpoint path/to/frozen_teacher.pth \
  --out runs/student \
  --device cpu
```

Replace all example paths with your own authorized local data and checkpoint paths. No commercial training, validation, or test data are distributed with this repository.

## Evaluate predictions and uncertainty

The evaluation utilities operate on prediction records generated from your own data.

For per-ROI relative-depth metrics:

```bash
python -m hed3d.evaluate relative path/to/predictions.csv
```

For independently calibrated physical-distance evaluation:

```bash
python -m hed3d.evaluate metric path/to/calibration.csv path/to/test.csv
```

To generate per-ROI predictions for uncertainty analysis, use `hed3d.score_student` with your own manifest and ROI images. For the manuscript's held-out evaluation, the expected sample count is 302:

```bash
python -m hed3d.score_student path/to/heldout_manifest.csv \
  --root path/to/roi_images \
  --out runs/evaluation/predictions.csv \
  --expected-count 302

python -m hed3d.reliability runs/evaluation/predictions.csv \
  --out runs/evaluation/reliability \
  --expected-count 302
```

The prediction workflow exports the fused prediction, both depth-head estimates, NIG parameters, SoftGate score and weight, theoretical predictive standard deviation, and dual-head disagreement. The reliability workflow computes depth errors, high-error labels, AUROC, average precision, Pearson correlations, and risk–coverage data for the available uncertainty scores.

The manuscript's 302-image evaluation requires the corresponding private ROI images and labels, or the original per-image prediction records. The commands above are workflow templates and can only reproduce the reported results when run with the same authorized evaluation data and protocol. The exact input formats and evaluation definitions are described in [`docs/reliability.md`](docs/reliability.md).

## Reproduce the published Figure 11 plot

The aggregate risk–coverage values used for the manuscript's Figure 11 are provided in `paper/figure11_risk_coverage.csv`. The plot can be regenerated without access to the original images:

```bash
python -m hed3d.plot_risk \
  paper/figure11_risk_coverage.csv \
  --out runs/paper/Figure11
```

This reproduces the plot from the reported aggregate values; it does not independently reproduce the underlying 302-image predictions or recompute the original metrics.

The aggregate Table 10 values are provided in `paper/table10_reported.csv` for reference.

## Scope and limitations

- HED-3D predicts an ROI-level normalized relative-depth scalar. It does not directly predict a dense pixel-wise depth map.
- Teacher-derived dense depth maps may be used for pseudo-label preparation and for the described thin-ring geometric analyses. Such visualizations and point sets should not be presented as dense depth predictions produced by the student network.
- Physical distance is obtained through a separate calibration procedure; it is not the direct output of the relative-depth student.
- The reported uncertainty-reliability metrics assess the ranking of prediction errors relative to the teacher-derived pseudo-labels. They should not be interpreted as proof of calibrated uncertainty for ground-truth metric distance.
- The reported latency measurements were obtained on the hardware platform specified in the manuscript. They should not be assumed to transfer unchanged to other GPUs, embedded devices, or robot controllers.
- The current work does not claim closed-loop robotic alignment or autonomous filling based on visual feedback. Such validation remains future work.

## Third-party components and checkpoint terms

HED-3D uses third-party software and model components, including PyTorch, torchvision, Ultralytics, and Depth Anything V2. Their respective licenses and usage terms apply. External repositories and checkpoints are not relicensed by this project.

The author-provided checkpoint in `weights/` is documented separately in [`weights/README.md`](weights/README.md). Checkpoint availability does not grant rights to the underlying commercial dataset or third-party model assets. Users are responsible for complying with all applicable licenses, permissions, and confidentiality obligations.

## Citation

If you use this implementation in academic work, please cite the associated HED-3D manuscript using its final bibliographic information.
