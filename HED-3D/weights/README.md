# DualKD inference checkpoint

`dualkd_student.pth` contains the student and SoftGate tensors from the author-provided
`mobilenetv3_spout_depth_student_nig_dual_best.pth` used by the archived ROI validation
entry points. The export contains only these two state dictionaries. Tensor values
were checked for exact equality after serializing the export.

- Archived file SHA-256 `5ee204a5e0c84055712ede0248857ce4c79cb6da2b375c80cdfaef21dea2d125`
- Public export SHA-256 `01df2b5cf8ff6587655c6b8110398dee944c61ac59f90d8f67a09135fcdd99cb`

Use `--checkpoint` to select a different evaluated checkpoint when replicating a specific
run. The frozen distillation teacher, the fine-tuned YOLO detector, and the external
Depth Anything V2 ViT-L checkpoint are provided separately by their respective owners.
