# Evidential model and evaluation protocol

## Comment 2 equations and implementation

The evidential head predicts $(\mu_{\mathrm{nig}}, \nu, \alpha, \beta_c)$ with $\nu>0$, $\alpha>1$, and $\beta_c>0$. The effective scale is $\beta_c=(1+\nu)\beta_{\mathrm{NIG}}$. The conventional marginal predictive distribution is Student-t with $2\alpha$ degrees of freedom and scale squared $\beta_c/(\alpha\nu)$. This gives $2\beta_{\mathrm{NIG}}(1+\nu)=2\beta_c$ in the predictive NLL. The manuscript's Eq. (12) and `nig_nll` use this effective scale and the original numerical stabilizers.

- Eq. (9a) is the theoretical predictive variance $V_{\mathrm{pred}}=\beta_c/[\nu(\alpha-1)]$. Its square root $\sigma_{\mathrm{pred}}$ is a post-hoc ranking score.
- Eq. (9b) uses $d=\max(\alpha-1,1.001)$ and $\epsilon=10^{-6}$. The stabilized SoftGate input is $u_{\mathrm{gate}}=\sqrt{\max(\beta_c/d+\beta_c/(\nu d+\epsilon),\epsilon)}$. It also supplies the ranking and score-distillation losses.
- Eq. (10) takes $w_{\mathrm{gate}}$ from the trained gate and computes $\mu_f=w_{\mathrm{gate}}\mu_{\mathrm{nig}}+(1-w_{\mathrm{gate}})\mu_{\mathrm{det}}$. The gate continues to receive $u_{\mathrm{gate}}$ during the reliability evaluation.

`hed3d/student.py` exposes the loss, both uncertainty calculations, and the shared forward path. `hed3d/train_student.py` implements the objective from Eqs. (11)–(19) using the checkpoint-compatible network. Its portable manifest interface checks training/validation separation and selects the output checkpoint by validation MAE. The archived training driver selected its best file by training objective. Recreating the original training run also requires its data and run configuration.

Depth Anything V2 supplies dense relative maps for the ring-aggregated labels. A separate pretrained dual-head MobileNetV3-Large with SoftGate supplies fused scalar depth, score, and gate-weight targets when `--kd-checkpoint` is passed. The checkpoint supplied to that option must come from training data only. It is loaded in evaluation mode with gradients disabled.

## Table 10 and Figure 11 from saved records

1. Use one row per valid thin-ring ROI. Supply `sample_id,image,Z_rel_cam` to `score_student`. Paths are relative to `--root`. A repeated ID, repeated image, or missing file fails the evaluation.
2. For each fused prediction, compute $e_i=\lvert\mu_{f,i}-y_i^*\rvert$. Label errors at or above the linearly interpolated 80th percentile as high error.
3. Calculate AUROC and average precision for $u_{\mathrm{gate}}$, $\sigma_{\mathrm{pred}}$, and absolute head disagreement on the same labels. Report the observed positive fraction as the chance AP. Pearson correlations use the absolute errors.
4. For each nominal coverage $c$ in 0.1 through 1.0, retain the lowest scoring $\max(1,\lfloor cN\rfloor)$ samples. Preserve input order for equal scores. Report the MAE and actual coverage for the retained prefix. Both curves include all predictions at full coverage.

Saved prediction records can bypass inference. The required CSV header is:

```csv
sample_id,target,prediction,u_gate,sigma_pred,disagreement
```

`target` and `prediction` are normalized relative depth. Each prediction must be the original fused output. Score entries are nonnegative. Preserve full precision when exporting records. For multiple ablation variants, run the relative-depth metrics on the same ordered sample IDs and the same target column. Threshold accuracy uses strict less-than at 0.05, 0.07, and 0.10.

For the independent physical-distance check, both calibration and test CSVs use `sample_id,relative_depth,distance_mm`. The affine fit uses the calibration rows only. The paper evaluates a 300–400 mm working range on its measured test set. Reliability metrics quantify errors against teacher pseudo-labels. Closed-loop robotic validation is outside the scope of these experiments.

## Included paper values and data access

`paper/table10_reported.csv` and `paper/figure11_risk_coverage.csv` are aggregate values transcribed from the revised paper, response, and plotting data. The paper lists 302 held-out images, 61 high-error cases, an error threshold of 0.014317, and a full-coverage MAE of 0.01055. The underlying 302 images and per-image measurements are required to independently recompute those reported metrics. The example produces separate results from synthetic inputs.
