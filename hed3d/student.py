"""DualKD architecture and checkpoint-compatible SoftGate."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from PIL import Image

EPS = 1e-6
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

class SpoutDepthStudentDual(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_large(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_feats = backbone.classifier[0].in_features
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_feats, 256),
            nn.ReLU(inplace=True),
        )
        self.head_nig = nn.Linear(256, 4)  # μ_nig_raw, v_raw, α_raw, β_raw
        self.head_det = nn.Linear(256, 1)  # μ_det_raw

    def forward(self, x):
        x = self.pool(self.features(x))            # [B,C,1,1]
        feat = self.fc(x)                          # [B,256]
        nig_o = self.head_nig(feat)                # [B,4]
        mu_nig_raw, v_raw, a_raw, b_raw = nig_o[:,0], nig_o[:,1], nig_o[:,2], nig_o[:,3]
        mu_nig = torch.sigmoid(mu_nig_raw)
        v      = F.softplus(v_raw) + 1e-3
        alpha  = F.softplus(a_raw) + 1.0 + 1e-3
        beta   = F.softplus(b_raw) + 1e-3
        mu_det_raw = self.head_det(feat).squeeze(1)   # [B]
        mu_det     = torch.sigmoid(mu_det_raw)
        return mu_nig, v, alpha, beta, mu_det

# 软门控（鲁棒版：统一 reshape 为一维）
class SoftGate(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)  # 输出 logit
        )
    def forward(self, mu_det, mu_nig, sigma_tot):
        # 确保一维 [B]
        mu_det   = mu_det.reshape(-1)
        mu_nig   = mu_nig.reshape(-1)
        sigma_tot= sigma_tot.reshape(-1)
        feats = torch.stack(
            [mu_det, mu_nig, sigma_tot, torch.abs(mu_det - mu_nig)],
            dim=-1
        )  # [B,4]
        logit = self.mlp(feats)        # [B,1]
        w = torch.sigmoid(logit)       # NIG 权重
        return logit, w


def nig_nll(y, mu, v, alpha, beta_c):
    """Equation (12), with beta_c = (1 + nu) * beta_NIG."""
    two_beta = 2.0 * beta_c
    res2 = (y - mu) ** 2
    term1 = 0.5 * torch.log(math.pi / (v + EPS))
    term2 = - alpha * torch.log(two_beta + EPS)
    term3 = (alpha + 0.5) * torch.log(v * res2 + two_beta + EPS)
    term4 = torch.lgamma(alpha + EPS) - torch.lgamma(alpha + 0.5 + EPS)
    return term1 + term2 + term3 + term4

def nig_evidence_regularizer(y, mu, v, alpha, coeff=5e-3):
    return coeff * (torch.abs(y - mu) * (2.0 * v + alpha)).mean()

def softgate_score(v, alpha, beta_c):
    """Equation (9b), the stabilized score used by the trained gate."""
    denom = (alpha - 1.0).clamp(min=1.001)
    var_alea = beta_c / denom
    var_epi  = beta_c / (v * denom + EPS)
    var_tot  = (var_alea + var_epi).clamp(min=EPS)
    return torch.sqrt(var_tot)


def nig_predictive_variance(v, alpha, beta_c):
    """Equation (9a), before the square root. Valid for alpha > 1."""
    return beta_c / (v * (alpha - 1.0))


def roi_outputs(student, gate, images):
    mu_nig, v, alpha, beta_c, mu_det = student(images)
    u_gate = softgate_score(v, alpha, beta_c)
    logits, weight = gate(mu_det, mu_nig, u_gate)
    weight = weight.flatten()
    fused = weight * mu_nig + (1 - weight) * mu_det
    return mu_nig, v, alpha, beta_c, mu_det, u_gate, logits.flatten(), weight, fused


def load_student(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    student = SpoutDepthStudentDual()
    gate = SoftGate()
    # A plain state dict cannot reconstruct the SoftGate.
    student.load_state_dict(state["student_state"])
    gate.load_state_dict(state["gate_state"])
    return student.to(device).eval(), gate.to(device).eval()


def evaluation_transform():
    return transforms.Compose([transforms.Resize((434, 434)), transforms.ToTensor(), NORMALIZE])


@torch.inference_mode()
def predict(student, gate, roi, device):
    roi = evaluation_transform()(roi.convert("RGB")).unsqueeze(0).to(device)
    mu_nig, v, alpha, beta_c, mu_det, u_gate, _, weight, fused = roi_outputs(student, gate, roi)
    variance = nig_predictive_variance(v, alpha, beta_c)
    return {
        "Z_rel_cam": fused.item(), "mu_nig": mu_nig.item(), "mu_det": mu_det.item(),
        "nu": v.item(), "alpha": alpha.item(), "beta_c": beta_c.item(),
        "u_gate": u_gate.item(), "w_gate": weight.item(),
        "predictive_variance": variance.item(), "sigma_pred": variance.sqrt().item(),
    }
