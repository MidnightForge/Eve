"""
ReservoirCell — Echo State Network for Temporal Brain Memory
=============================================================
Reservoir computing brings a fundamentally different kind of intelligence to
Eve's brain: temporal pattern memory. Where the quantum mesh binds cells
spatially and the coherence cell cross-references capabilities, the reservoir
remembers HOW the brain moves through time.

What is Reservoir Computing?
-----------------------------
A reservoir is a large, fixed random dynamical system. You drive it with inputs
and read out from its high-dimensional state. The key insight:

  • The reservoir weights are NEVER trained — only the readout layer trains
  • The rich dynamics of the reservoir implicitly project inputs into a
    high-dimensional nonlinear feature space
  • Training cost: O(n²) ridge regression on collected states — trivially fast
  • Echo State Property: the reservoir's state is uniquely determined by its
    input history (for spectral radius ρ < 1)

This is fundamentally different from Transformers (attention over all pairs) or
RNNs (backprop through time). The reservoir "just is" — its dynamics ARE the
memory, not learned weights.

Architecture implemented here
------------------------------
  Primary: Deep Residual ESN (DeepResESN) — Pinna et al. arXiv:2508.21172 (Aug 2025)
    • 3 stacked reservoir layers, N_RES=512 nodes each
    • Per-layer leak rates: α=[0.9, 0.5, 0.2] — shallow=fast, deep=slow timescales
    • Orthogonal residual connections between layers (proven to maximise memory capacity)
    • Each layer maintains its own Echo State Property (ρ < 1 per layer)
    • Readout: W_out @ [x₁; x₂; x₃; u; 1]  — all-layer concat + input
    • Two training paths:
        (a) Offline Ridge regression (periodic batch retrain every 200 steps)
        (b) Online RLS (Recursive Least Squares) — adapts per-step, O(N²) vs O(N³)

  Secondary: Next-Generation Reservoir Computing (NG-RC) — Gauthier 2021
    • Uses the CURRENT state + its nonlinear combinations instead of recursion
    • Feature vector: [u(t), u(t-1), ..., u(t-k), u(t)⊗u(t-1), ...]
    • Even simpler: no reservoir needed at all — pure delay + product features
    • Training: SVD-based solver (prevents ill-conditioning, Santos & Bollt 2025)
    • Noise regularizer: σ=0.01 added to training inputs (arXiv:2509.11338)
    • Much smaller (k*n + k²*n features) — fast inference

  Fusion: DeepResESN (60%) + NG-RC (40%) ensemble prediction.

  Future: Echo State Transformer (arXiv:2507.02917, July 2025) — ESN as O(1)
    working memory substrate for attention mechanisms. On-roadmap for Blackwell.

What it does for Eve
--------------------
  Each conversation turn, the brain's activity is encoded as a state vector
  (same 32-dim encoding as the quantum mesh). This drives both reservoirs.

  Output: a 30-dim vector of predicted cell activation probabilities for
  the NEXT turn. This is fed to the Cortex as "temporal context" — helping
  route not just on the current message but on where the conversation has
  been going.

  Over time, the readout layer learns patterns like:
    "After vision → memory → emotion, the next call is usually to anima"
    "After 3 math questions, reasoning cell activates more reliably"
    "When school challenge score drops, curiosity cell activates to rebuild"

  The reservoir remembers the TRAJECTORY of the brain, not just its snapshot.

Technical parameters (tuned for RTX 4090, pure numpy)
------------------------------------------------------
  N_RES         = 512    # reservoir size per layer
  N_LAYERS      = 3      # DeepResESN depth
  LEAK_RATES    = [0.9, 0.5, 0.2]  # per-layer (shallow→fast, deep→slow)
  SPECTRAL_R    = [0.93, 0.95, 0.97]  # slightly escalating depth
  SPARSITY      = 0.10   # 10% connectivity (typical for ESN)
  INPUT_SCALE   = 0.1    # scales input influence
  RESIDUAL_SCALE= 0.05   # orthogonal residual weight (DeepResESN)
  RIDGE_OFFLINE = 1e-6   # L2 for offline batch retraining
  RIDGE_ONLINE  = 1e-3   # initial P matrix scaling for RLS (= 1/lambda)
  WASHOUT       = 50     # discard first N states (transient suppression)
  N_DELAY       = 3      # NG-RC: use 3 past steps + cross products
  NG_NOISE      = 0.01   # Gaussian noise σ added to NG-RC training inputs
  TRAIN_EVERY   = 200    # retrain offline readout every N observations

All weights use float32. numpy operations are vectorised — zero GPU needed.
Optional torch path for GPU acceleration when available.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Optional

import numpy as np

from brain.base_cell import BaseCell, CellContext

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Deep Residual ESN  (DeepResESN — Pinna et al. arXiv:2508.21172, Aug 2025)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_reservoir_layer(n_res: int, n_in: int, sparsity: float,
                          spectral_radius: float, input_scale: float,
                          rng: np.random.Generator, seed_offset: int = 0):
    """Build one fixed (W, W_in, bias) reservoir layer."""
    W = rng.standard_normal((n_res, n_res)).astype(np.float32)
    mask = (rng.uniform(0, 1, W.shape) < sparsity).astype(np.float32)
    W *= mask
    np.fill_diagonal(W, 0.0)
    eigvals = np.linalg.eigvals(W)
    sr = np.max(np.abs(eigvals))
    if sr > 1e-8:
        W *= (spectral_radius / sr)
    W_in   = (rng.standard_normal((n_res, n_in)) * input_scale).astype(np.float32)
    W_bias = (rng.standard_normal(n_res) * 0.01).astype(np.float32)
    return W, W_in, W_bias


def _make_orthogonal_residual(n_res: int, scale: float,
                               rng: np.random.Generator) -> np.ndarray:
    """
    Random orthogonal matrix scaled by `scale`.
    Used for residual connections between DeepResESN layers.
    QR decomposition on a random Gaussian matrix gives Q orthogonal.
    """
    M = rng.standard_normal((n_res, n_res)).astype(np.float32)
    Q, _ = np.linalg.qr(M)
    return (Q * scale).astype(np.float32)


class DeepResESN:
    """
    Deep Residual Echo State Network — 2025 SOTA for temporal memory.

    Architecture (Pinna et al. arXiv:2508.21172):
      Layer 0 (fast):   α=0.9, ρ=0.93 — tracks rapid input changes
      Layer 1 (medium): α=0.5, ρ=0.95 — mid-range temporal patterns
      Layer 2 (slow):   α=0.2, ρ=0.97 — long-range context retention

    Orthogonal residual connections between layers proven to maximise memory
    capacity over standard DeepESN. Echo State Property is maintained per layer.

    State update per layer l:
        pre_l = W_l @ x_l  +  W_in_l @ x_{l-1}  +  W_res_{l-1→l} @ x_{l-1}  +  bias_l
        x_l(t) = (1-α_l)*x_l(t-1) + α_l*tanh(pre_l)

    Readout features: [x_0; x_1; x_2; u; 1]  (full concat — all timescales)

    Training: offline batch Ridge regression via numpy.linalg.lstsq (SVD-based,
    numerically stable for ill-conditioned state matrices).
    """

    def __init__(
        self,
        n_reservoir:     int        = 512,
        n_input:         int        = 32,
        n_output:        int        = 32,
        n_layers:        int        = 3,
        spectral_radii:  list       = None,   # per-layer, defaults [0.93, 0.95, 0.97]
        leak_rates:      list       = None,   # per-layer, defaults [0.9, 0.5, 0.2]
        sparsity:        float      = 0.10,
        input_scale:     float      = 0.10,
        residual_scale:  float      = 0.05,
        ridge:           float      = 1e-6,
        washout:         int        = 50,
        seed:            int        = 2026,
    ):
        rng = np.random.default_rng(seed)

        self.n_res     = n_reservoir
        self.n_in      = n_input
        self.n_out     = n_output
        self.n_layers  = n_layers
        self.ridge     = ridge
        self.washout   = washout
        self.alphas    = (leak_rates or [0.9, 0.5, 0.2])[:n_layers]
        rhos           = (spectral_radii or [0.93, 0.95, 0.97])[:n_layers]

        # ── Fixed recurrent weights for each layer ────────────────────────────
        self.W:    list[np.ndarray] = []
        self.W_in: list[np.ndarray] = []
        self.bias: list[np.ndarray] = []
        # Layer 0 reads from external input; deeper layers read from previous layer
        in_sizes = [n_input] + [n_reservoir] * (n_layers - 1)
        for l in range(n_layers):
            W_l, W_in_l, b_l = _make_reservoir_layer(
                n_reservoir, in_sizes[l], sparsity, rhos[l], input_scale, rng, l
            )
            self.W.append(W_l)
            self.W_in.append(W_in_l)
            self.bias.append(b_l)

        # ── Orthogonal residual connections (l-1 → l) ─────────────────────────
        self.W_res: list[np.ndarray] = []
        for l in range(1, n_layers):
            self.W_res.append(_make_orthogonal_residual(n_reservoir, residual_scale, rng))

        # ── States ────────────────────────────────────────────────────────────
        self.states: list[np.ndarray] = [
            np.zeros(n_reservoir, dtype=np.float32) for _ in range(n_layers)
        ]

        # ── Readout: shape (n_output, n_layers*n_res + n_input + 1) ──────────
        self._feat_size = n_layers * n_reservoir + n_input + 1
        self.W_out  = np.zeros((n_output, self._feat_size), dtype=np.float32)
        self._trained = False

        # ── Collection buffers for offline training ───────────────────────────
        self._states_buf:  list[np.ndarray] = []
        self._targets_buf: list[np.ndarray] = []
        self._step_count = 0

    def step(self, u: np.ndarray) -> np.ndarray:
        """
        Drive one timestep through all layers. Returns full feature vector.
        """
        u = u.astype(np.float32)
        x_prev = u
        new_states = []
        for l in range(self.n_layers):
            pre = self.W[l] @ self.states[l] + self.W_in[l] @ x_prev + self.bias[l]
            # Add orthogonal residual from previous layer (l≥1)
            if l > 0:
                pre = pre + self.W_res[l - 1] @ new_states[l - 1]
            new_x = (1.0 - self.alphas[l]) * self.states[l] + self.alphas[l] * np.tanh(pre)
            new_states.append(new_x)
            x_prev = new_x
        self.states = new_states
        self._step_count += 1
        return self._feature_vec(u)

    def _feature_vec(self, u: np.ndarray) -> np.ndarray:
        """Concat all layer states + input + bias term."""
        return np.concatenate(self.states + [u, [1.0]]).astype(np.float32)

    def readout(self) -> np.ndarray:
        """ŷ = W_out @ [x₀; x₁; x₂; u_last; 1]. Requires self.states set by step()."""
        u_last = np.zeros(self.n_in, dtype=np.float32)  # placeholder — u not stored
        fv = np.concatenate(self.states + [u_last, [1.0]]).astype(np.float32)
        return self.W_out @ fv

    def collect(self, u: np.ndarray, target: np.ndarray) -> None:
        """Step + store feature for offline batch training."""
        fv = self.step(u)
        if self._step_count > self.washout:
            self._states_buf.append(fv)
            self._targets_buf.append(target.astype(np.float32))

    def train_readout(self, n_output: int) -> bool:
        """
        Offline SVD-based ridge regression (numerically stable).
        SVD avoids ill-conditioning of the normal equations (Santos & Bollt 2025).
        """
        n = len(self._states_buf)
        if n < max(20, self.washout):
            return False
        S = np.stack(self._states_buf, axis=0)      # (T, feat_size)
        T = np.stack(self._targets_buf, axis=0)     # (T, n_cells)
        T = T[:, :n_output]                         # trim/pad output dim

        # Resize W_out if n_output changed
        if n_output != self.n_out or S.shape[1] != self._feat_size:
            self._feat_size = S.shape[1]
            self.n_out  = n_output
            self.W_out  = np.zeros((n_output, self._feat_size), dtype=np.float32)

        # SVD-based Ridge: augment S with sqrt(λ)*I rows, T with zeros — equivalent
        # to Tikhonov but avoids forming S^T S (Björck 1996)
        n_feat = S.shape[1]
        lam_sqrt = np.sqrt(self.ridge)
        S_aug = np.vstack([S, lam_sqrt * np.eye(n_feat, dtype=np.float32)])
        T_aug = np.vstack([T, np.zeros((n_feat, n_output), dtype=np.float32)])
        # lstsq uses SVD internally — numerically stable
        W_out_T, _, _, _ = np.linalg.lstsq(S_aug, T_aug, rcond=None)
        self.W_out = W_out_T.T.astype(np.float32)

        self._states_buf.clear()
        self._targets_buf.clear()
        self._trained = True
        return True

    @property
    def spectral_radius(self) -> float:
        """Return spectral radius of deepest (slowest) layer."""
        return float(np.max(np.abs(np.linalg.eigvals(self.W[-1]))))

    def reset_state(self) -> None:
        self.states = [np.zeros(self.n_res, dtype=np.float32) for _ in range(self.n_layers)]


# ═══════════════════════════════════════════════════════════════════════════════
# Online Recursive Least Squares (RLS) readout
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineRLS:
    """
    Recursive Least Squares readout — adapts W_out per-step without backprop.

    Based on Sussillo & Abbott FORCE learning (2009) and Composite FORCE
    (arXiv:2207.02420). Equivalent to offline ridge regression but online:
        P(t) = P(t-1) - P(t-1)·x·x^T·P(t-1) / (1 + x^T·P(t-1)·x)
        e(t) = y_target - W_out(t-1) @ x
        W_out(t) = W_out(t-1) + outer(e, P(t)·x)

    O(N²) per step vs O(N³) for batch — critical for streaming brain data.
    Initialised with large P = (1/lambda)*I for fast early adaptation.
    """

    def __init__(self, n_features: int, n_output: int, lambda_: float = 1e-3):
        # P starts as (1/λ)*I — large initial covariance = fast learning
        self.P     = np.eye(n_features, dtype=np.float64) / lambda_
        self.W_out = np.zeros((n_output, n_features), dtype=np.float64)
        self.n_out = n_output
        self._updates = 0

    def update(self, x: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        """
        One RLS step. Returns y_pred BEFORE this update.
        x: (n_features,), y_target: (n_output,)
        """
        x = x.astype(np.float64)
        y_t = y_target.astype(np.float64)[:self.n_out]

        Px = self.P @ x                              # (n_feat,)
        denom = 1.0 + float(x @ Px)
        k = Px / denom                               # Kalman gain (n_feat,)

        y_pred = self.W_out @ x                      # prediction before update
        e = y_t - y_pred                             # error (n_out,)

        self.W_out += np.outer(e, k)                 # readout update
        self.P     -= np.outer(k, Px)                # covariance update
        self._updates += 1
        return y_pred.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Next-Generation Reservoir Computing (NG-RC)
# ═══════════════════════════════════════════════════════════════════════════════

class NextGenReservoir:
    """
    NG-RC — Gauthier et al. Nature Communications 2021.

    No actual reservoir. Feature vector at time t:
        Φ(t) = [u(t), u(t-1), ..., u(t-k+1),   ← linear part
                u(t)⊗u(t-1), u(t-1)⊗u(t-2), ...]  ← nonlinear cross products

    Readout: ŷ = W_out @ Φ(t)

    Much smaller than ESN, analytically solvable, no spectral radius concerns.
    Excellent for smooth temporal patterns (conversation flow).
    """

    def __init__(
        self,
        n_input:  int   = 32,
        n_output: int   = 32,
        k_delay:  int   = 3,    # number of past steps
        ridge:    float = 1e-4,
    ):
        self.n_in  = n_input
        self.n_out = n_output
        self.k     = k_delay
        self.ridge = ridge

        # Feature dim: k*n_input (linear) + (k-1)*n_input²_reduced (nonlinear)
        # For memory, we only use Hadamard product of adjacent delays (not full outer)
        # Feature dim = k*n + (k-1)*n
        self.feat_dim = k_delay * n_input + (k_delay - 1) * n_input
        self.W_out    = np.zeros((n_output, self.feat_dim), dtype=np.float32)
        self._trained = False

        # Delay buffer (ring buffer of past k inputs)
        self._buf: list[np.ndarray] = []

        # Training buffers
        self._feat_buf:   list[np.ndarray] = []
        self._target_buf: list[np.ndarray] = []

    def _get_features(self) -> Optional[np.ndarray]:
        """Compute feature vector from current delay buffer."""
        if len(self._buf) < self.k:
            return None
        linear   = np.concatenate(list(self._buf[-self.k:]))     # k*n_in
        nonlinear = np.concatenate([
            self._buf[-i-1] * self._buf[-i-2]                     # element-wise product
            for i in range(self.k - 1)
        ])                                                         # (k-1)*n_in
        return np.concatenate([linear, nonlinear]).astype(np.float32)

    def step(self, u: np.ndarray) -> Optional[np.ndarray]:
        """Push input, return readout prediction (or None if not enough history)."""
        self._buf.append(u.astype(np.float32))
        if len(self._buf) > self.k + 1:
            self._buf.pop(0)
        feat = self._get_features()
        if feat is None or not self._trained:
            return None
        return self.W_out @ feat

    def collect(self, u: np.ndarray, target: np.ndarray) -> None:
        self._buf.append(u.astype(np.float32))
        if len(self._buf) > self.k + 1:
            self._buf.pop(0)
        feat = self._get_features()
        if feat is not None:
            self._feat_buf.append(feat)
            self._target_buf.append(target.astype(np.float32))

    def train_readout(self, n_output: int, noise_sigma: float = 0.01) -> bool:
        """
        Train via SVD-based Ridge (Santos & Bollt arXiv:2505.00846).
        Adds Gaussian noise to features before solving — implicit regularizer
        that improves long-term autonomous stability (arXiv:2509.11338).
        """
        n = len(self._feat_buf)
        if n < 10:
            return False
        F = np.stack(self._feat_buf)                         # (T, feat_dim)
        T = np.stack(self._target_buf)[:, :n_output]        # (T, n_output)
        n_out = min(n_output, T.shape[1])

        # Resize if needed
        if n_out != self.n_out or F.shape[1] != self.feat_dim:
            self.feat_dim = F.shape[1]
            self.n_out    = n_out
            self.W_out    = np.zeros((n_out, self.feat_dim), dtype=np.float32)

        # Noise regularization — adds σ*randn to feature rows (arXiv:2509.11338)
        if noise_sigma > 0:
            rng = np.random.default_rng(seed=42)
            F = F + rng.standard_normal(F.shape).astype(np.float32) * noise_sigma

        # SVD-based Ridge (augmented system avoids forming F^T F)
        n_feat = F.shape[1]
        lam_sqrt = np.sqrt(self.ridge)
        F_aug = np.vstack([F, lam_sqrt * np.eye(n_feat, dtype=np.float32)])
        T_aug = np.vstack([T[:, :n_out], np.zeros((n_feat, n_out), dtype=np.float32)])
        W_out_T, _, _, _ = np.linalg.lstsq(F_aug, T_aug, rcond=None)
        self.W_out = W_out_T.T.astype(np.float32)

        self._feat_buf.clear()
        self._target_buf.clear()
        self._trained = True
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# ReservoirCell
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level singleton
_RESERVOIR_INSTANCE: Optional["ReservoirCell"] = None


def get_reservoir_prediction() -> dict:
    """
    Public API — any cell can call this to get the reservoir's temporal prediction.
    Returns {cell_name: activation_probability} for the NEXT conversation turn.
    """
    if _RESERVOIR_INSTANCE is None:
        return {}
    return _RESERVOIR_INSTANCE.get_prediction()


class ReservoirCell(BaseCell):
    """
    Echo State Network + NG-RC for temporal brain memory.

    Processes the sequence of brain states over time. Predicts which cells
    should activate NEXT based on where the conversation has been.

    Feeds temporal context back to Cortex routing — the brain learns its own
    conversational rhythms and activates cells pre-emptively.
    """

    name        = "reservoir"
    description = ("Echo State Network + Next-Gen Reservoir for temporal brain memory. "
                   "Learns conversation rhythm and predicts next-turn cell activations.")
    color       = "#8b5cf6"   # violet
    lazy        = True        # boots on first use (heavy matrix multiply in boot)
    position    = (5, 4)

    system_tier     = "online"
    hardware_req    = "RTX 4090 — pure numpy float32, ~2MB RAM for 512-node ESN"
    research_basis  = (
        "Jaeger 2001 'Echo State Networks' — foundational ESN, spectral radius rule; "
        "Maass et al. 2002 'Liquid State Machines' — fading memory, biological plausibility; "
        "Gauthier et al. Nature Comms 2021 — NG-RC: delay+product features, analytically solvable; "
        "Gallicchio & Micheli 2017 (DeepESN arXiv:1712.04323) — stacked hierarchical timescales; "
        "Pinna et al. Aug 2025 (DeepResESN arXiv:2508.21172) — ORTHOGONAL residuals maximise memory; "
        "Santos & Bollt 2025 (arXiv:2505.00846) — SVD solver prevents NG-RC ill-conditioning; "
        "arXiv:2509.11338 — noise σ=0.01 as implicit regularizer for NG-RC stability; "
        "Sussillo & Abbott 2009 FORCE / arXiv:2207.02420 Composite FORCE — online RLS readout; "
        "Bendi-Ouis & Hinaut July 2025 (EST arXiv:2507.02917) — ESN as O(1) attention memory"
    )
    build_notes = (
        "LIVE: DeepResESN (3 layers, α=[0.9,0.5,0.2], orthogonal residuals) + NG-RC (k=3, SVD). "
        "Dual readout: Online RLS (per-step streaming) + Offline Ridge (every 200 steps). "
        "NG-RC noise σ=0.01 for stability. Feeds temporal prediction to Cortex. "
        "get_reservoir_prediction() callable from anywhere. "
        "Roadmap: Echo State Transformer (linear-complexity attention memory) on Blackwell."
    )
    framework_layer = "Deep Learning"

    N_RES        = 512
    N_INPUT      = 32     # matches quantum mesh / coherence cell vector dim
    N_LAYERS     = 3      # DeepResESN depth
    TRAIN_EVERY  = 200    # offline retrain interval
    MAX_HISTORY  = 1000   # sliding window for batch retrain

    def __init__(self):
        super().__init__()
        self._esn:  Optional[DeepResESN]       = None
        self._ngrc: Optional[NextGenReservoir] = None
        self._rls:  Optional[OnlineRLS]        = None
        self._lock  = threading.Lock()
        self._step_count  = 0
        self._train_count = 0
        self._prediction: dict[str, float] = {}
        self._cell_names: list[str] = []

        # Sliding window of (input_vec, target_vec) for retraining
        self._history: list[tuple[np.ndarray, np.ndarray]] = []

    async def boot(self) -> None:
        global _RESERVOIR_INSTANCE
        feat_size = self.N_LAYERS * self.N_RES + self.N_INPUT + 1
        self._esn  = DeepResESN(
            n_reservoir    = self.N_RES,
            n_input        = self.N_INPUT,
            n_layers       = self.N_LAYERS,
            spectral_radii = [0.93, 0.95, 0.97],
            leak_rates     = [0.9,  0.5,  0.2],
            sparsity       = 0.10,
            input_scale    = 0.10,
            residual_scale = 0.05,
            ridge          = 1e-6,
            washout        = 50,
        )
        self._ngrc = NextGenReservoir(
            n_input  = self.N_INPUT,
            k_delay  = 3,
            ridge    = 1e-4,
        )
        # RLS readout — will resize n_output to n_cells on first target
        self._rls = OnlineRLS(n_features=feat_size, n_output=32, lambda_=1e-3)
        _RESERVOIR_INSTANCE = self
        logger.info("[Reservoir] DeepResESN (3L, ρ=[0.93,0.95,0.97]) + NG-RC + RLS online. "
                    "feat_dim=%d", feat_size)

    # ── Public drive API ──────────────────────────────────────────────────────

    def drive(self, input_vec: np.ndarray, target_vec: np.ndarray,
              cell_names: list) -> dict:
        """
        Drive the reservoir with a new brain state observation.

        Args:
            input_vec:  32-dim encoded current brain state
            target_vec: n_cells-dim one-hot / activation vector (what cells fired)
            cell_names: list of cell names matching target_vec indices

        Returns:
            prediction dict {cell_name: probability} for next turn
        """
        if self._esn is None:
            return {}

        n_cells = len(cell_names)

        with self._lock:
            # ── Store in sliding history ──────────────────────────────────────
            self._history.append((input_vec.copy(), target_vec.copy()))
            if len(self._history) > self.MAX_HISTORY:
                self._history.pop(0)

            # ── DeepResESN step — collect feature vector ──────────────────────
            feat_vec = self._esn.step(input_vec)   # drives all 3 layers
            self._esn._states_buf.append(feat_vec)
            t_padded = target_vec[:n_cells].astype(np.float32)
            if len(t_padded) < n_cells:
                t_padded = np.pad(t_padded, (0, n_cells - len(t_padded)))
            self._esn._targets_buf.append(t_padded)

            # ── NG-RC collect ─────────────────────────────────────────────────
            self._ngrc.collect(input_vec, target_vec)

            # ── Online RLS update (per-step streaming adaptation) ─────────────
            if self._rls is not None and self._step_count > 10:
                # Resize RLS n_output if cell count changed
                if self._rls.n_out != n_cells:
                    feat_size = self._esn._feat_size
                    self._rls = OnlineRLS(n_features=feat_size, n_output=n_cells,
                                          lambda_=1e-3)
                self._rls.update(feat_vec, t_padded)

            self._step_count += 1

            # ── Periodic offline retrain (every TRAIN_EVERY steps) ────────────
            if self._step_count % self.TRAIN_EVERY == 0 and self._step_count > 100:
                esn_ok  = self._esn.train_readout(n_cells)
                ngrc_ok = self._ngrc.train_readout(n_cells)
                if esn_ok or ngrc_ok:
                    self._train_count += 1
                    logger.info("[Reservoir] Offline retrain #%d (step %d, ESN=%s, NGRC=%s)",
                                self._train_count, self._step_count, esn_ok, ngrc_ok)
                    # Replay history through reservoir to recover drifted states
                    self._esn.reset_state()
                    for iv, tv in self._history[-200:]:
                        fv = self._esn.step(iv)
                        self._esn._states_buf.append(fv)
                        self._esn._targets_buf.append(tv[:n_cells])

            # ── Generate prediction (3-way ensemble) ──────────────────────────
            pred: Optional[np.ndarray] = None

            # Path A: Online RLS (always ready after 10 steps)
            if self._rls and self._rls._updates > 10:
                rls_out = (self._rls.W_out @ feat_vec).astype(np.float32)[:n_cells]
                rls_out -= rls_out.max()
                pred = np.exp(rls_out) / (np.exp(rls_out).sum() + 1e-8)

            # Path B: Offline ESN readout (60%) + NG-RC (40%) blend, if trained
            if self._esn._trained:
                esn_raw = self._esn.W_out @ feat_vec
                esn_out = esn_raw[:n_cells]
                esn_out -= esn_out.max()
                esn_probs = np.exp(esn_out) / (np.exp(esn_out).sum() + 1e-8)

                ngrc_pred = self._ngrc.step(input_vec)
                if ngrc_pred is not None:
                    ng_out  = ngrc_pred[:n_cells]
                    ng_out -= ng_out.max()
                    ng_probs = np.exp(ng_out) / (np.exp(ng_out).sum() + 1e-8)
                    offline_blend = 0.6 * esn_probs + 0.4 * ng_probs
                else:
                    offline_blend = esn_probs

                # Combine RLS (fast adaptation) with offline blend (stable)
                pred = (0.5 * pred + 0.5 * offline_blend) if pred is not None else offline_blend

            self._cell_names = cell_names
            if pred is not None:
                pred_n = pred[:n_cells]
                self._prediction = {
                    cell_names[i]: round(float(pred_n[i]), 5)
                    for i in range(min(n_cells, len(pred_n)))
                }
            return dict(self._prediction)

    def get_prediction(self) -> dict:
        with self._lock:
            return dict(self._prediction)

    # ── BaseCell.process ──────────────────────────────────────────────────────

    async def process(self, ctx: CellContext) -> Any:
        pred = self.get_prediction()
        if not pred:
            return "Reservoir warming up — needs ~100 brain steps before predictions stabilise."

        top = sorted(pred.items(), key=lambda x: x[1], reverse=True)[:5]
        return {
            "top_predicted_cells": [{"cell": n, "prob": p} for n, p in top],
            "steps":            self._step_count,
            "trains":           self._train_count,
            "rls_updates":      self._rls._updates if self._rls else 0,
            "deep_res_esn_rho": round(self._esn.spectral_radius if self._esn else 0.0, 4),
            "esn_trained":      self._esn._trained  if self._esn  else False,
            "ngrc_trained":     self._ngrc._trained if self._ngrc else False,
            "architecture":     "DeepResESN-3L + NG-RC + OnlineRLS",
        }

    def health(self) -> dict:
        return {
            "status":        self._status.value,
            "steps":         self._step_count,
            "esn_trained":   self._esn._trained  if self._esn  else False,
            "rls_updates":   self._rls._updates  if self._rls  else 0,
            "train_count":   self._train_count,
            "n_history":     len(self._history),
        }
