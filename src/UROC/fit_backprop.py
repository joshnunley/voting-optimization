#!/usr/bin/env python3
"""
C(K, α) = K - a₁(K)·α² - a₂(K)·α

where a₁, a₂ interpolate linearly with K ∈ [1, 20]:
  a₁(K) = a₁s + (a₁e - a₁s)·(K-1)/19
  a₂(K) = a₂s + (a₂e - a₂s)·(K-1)/19

4 learnable params: a₁s, a₁e, a₂s, a₂e.
At α=0: C = K.  The correction is purely in α.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb

FIGDIR = "results/figures"
os.makedirs(FIGDIR, exist_ok=True)

METHODS = ["borda", "irv", "minimax", "plurality", "star", "total_score"]
METHOD_LABELS = {"borda": "Borda", "irv": "IRV", "minimax": "Minimax",
                 "plurality": "Plurality", "star": "STAR", "total_score": "Total Score"}
METHOD_COLORS = {"borda": "#377eb8", "irv": "#4daf4a", "minimax": "#984ea3",
                 "plurality": "#ff7f00", "star": "#f781bf", "total_score": "#999999"}

K_ARR = np.arange(1, 21, dtype=np.float32)
A_ARR = np.round(np.arange(0, 1.05, 0.05), 2).astype(np.float32)
nK, nA = len(K_ARR), len(A_ARR)


def load_data():
    d = np.load("results/direct_results.npz")
    all_m = list(d['methods'])
    keep = [all_m.index(m) for m in METHODS]
    return d['mean'][:, :, keep].astype(np.float32)


def compute_ranks(mean):
    nK, nA, nM = mean.shape
    ranks = np.zeros_like(mean)
    for i in range(nK):
        for j in range(nA):
            order = np.argsort(-mean[i, j, :])
            for r, idx in enumerate(order):
                ranks[i, j, idx] = r + 1
    return ranks


def build_tensors(ranks):
    Ks, alphas, rvecs = [], [], []
    for i, K in enumerate(K_ARR):
        for j, alpha in enumerate(A_ARR):
            Ks.append(K)
            alphas.append(alpha)
            rvecs.append(ranks[i, j, :])
    return (torch.tensor(Ks), torch.tensor(alphas),
            torch.tensor(np.array(rvecs)))


class ComplexityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Parametrize as (c, μ) with constraints:
        #   c > 0 via softplus
        #   μ_start > μ_end via: μ_s = 0.5 + gap, μ_e = 0.5 - gap  (shared gap)
        #   or just let both be free in [0.3, 0.7] and add penalty
        # c(K) = a · (K-1)²,  μ(K) = 1/2 + b/(K-1)
        # Just 2 free params: a, b
        self.raw_a = torch.nn.Parameter(torch.tensor(1.0))   # softplus → ~1.8
        self.raw_b = torch.nn.Parameter(torch.tensor(-0.5))  # softplus → ~0.4

    @property
    def a(self):
        return torch.nn.functional.softplus(self.raw_a)

    @property
    def b(self):
        return torch.nn.functional.softplus(self.raw_b)

    def c_of_K(self, K):
        """c(K) = a · (K-1)²"""
        return self.a * (torch.clamp(K - 1.0, min=0.0)) ** 2

    def mu_of_K(self, K):
        """μ(K) = 1/2 + b/(K-1)"""
        return 0.5 + self.b / torch.clamp(K - 1.0, min=1e-3)

    def get_params_at_K0(self, K0):
        with torch.no_grad():
            c = self.c_of_K(torch.tensor(float(K0))).item()
            mu = self.mu_of_K(torch.tensor(float(K0))).item()
        return c, mu


def loss_fn(model, Ks, alphas, rvecs, eps=0.5):
    """For each K0 in 1..20, compute the parabola, weight cells by
    Gaussian proximity, measure rank MSD among nearby cells."""
    n = len(Ks)
    nM = rvecs.shape[1]
    total_loss = torch.tensor(0.0)

    for K0 in range(1, 21):
        K0_t = torch.tensor(float(K0))
        c = model.c_of_K(K0_t)
        mu = model.mu_of_K(K0_t)

        # Distance of each cell to this parabola
        K_on_curve = c * (alphas - mu)**2 + K0_t
        dist = Ks - K_on_curve  # signed distance in K
        weights = torch.exp(-dist**2 / (2 * eps**2))

        # Weighted mean rank vector
        w_sum = weights.sum() + 1e-8
        mean_rank = (weights.unsqueeze(1) * rvecs).sum(dim=0) / w_sum

        # Weighted variance of rank vectors around the mean
        deviations = rvecs - mean_rank.unsqueeze(0)  # (n, nM)
        var = (weights.unsqueeze(1) * deviations**2).sum(dim=0) / w_sum  # (nM,)
        total_loss = total_loss + var.mean()

    return total_loss / 20.0


def fit(ranks, eps=0.5, lr=0.05, steps=3000):
    Ks, alphas, rvecs = build_tensors(ranks)
    model = ComplexityModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    best_loss = float('inf')
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model, Ks, alphas, rvecs, eps=eps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # NaN guard
        if torch.isnan(loss):
            model.load_state_dict(best_state)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr*0.5)
            continue
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (step + 1) % 200 == 0:
            with torch.no_grad():
                c1 = model.c_of_K(torch.tensor(1.0)).item()
                c10 = model.c_of_K(torch.tensor(10.0)).item()
                c20 = model.c_of_K(torch.tensor(20.0)).item()
                mu1 = model.mu_of_K(torch.tensor(1.0)).item()
                mu20 = model.mu_of_K(torch.tensor(20.0)).item()
                print(f"  step {step+1}: loss={loss.item():.4f}  "
                      f"c=[{c1:.1f},{c10:.1f},{c20:.1f}]  "
                      f"a={model.a.item():.3f} b={model.b.item():.3f}  "
                      f"μ=[{mu1:.3f},{mu20:.3f}]"
                      f"{'  *best*' if loss.item() <= best_loss else ''}",
                      flush=True)

    model.load_state_dict(best_state)
    return model, best_loss


def plot_results(mean, ranks, model):
    print(f"\nFitted (2 params):")
    print(f"  a = {model.a.item():.4f}")
    print(f"  b = {model.b.item():.4f}")
    print(f"  c(K) = {model.a.item():.4f} · (K-1)²")
    print(f"  μ(K) = 1/2 + {model.b.item():.4f} / (K-1)")

    print(f"\n  {'K₀':>3s}  {'c':>7s}  {'μ':>6s}  K(α=0)  K(α=.5)  K(α=1)")
    for K0 in [1, 5, 10, 15, 20]:
        c, mu = model.get_params_at_K0(K0)
        K_at_0 = c * mu**2 + K0
        K_at_half = c * (0.5 - mu)**2 + K0
        K_at_1 = c * (1 - mu)**2 + K0
        print(f"  {K0:3d}  {c:7.2f}  {mu:6.3f}  {K_at_0:6.1f}   {K_at_half:6.1f}   {K_at_1:6.1f}")

    # ── 1. a₁(K), μ(K) ──
    Ks = np.linspace(1, 20, 100)
    with torch.no_grad():
        c_p = np.array([model.c_of_K(torch.tensor(float(K))).item() for K in Ks])
        mu_p = np.array([model.mu_of_K(torch.tensor(float(K))).item() for K in Ks])
    a1_p = c_p

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(Ks, a1_p, 'b-', lw=2); ax1.set_xlabel("K"); ax1.set_ylabel("a₁")
    ax1.set_title("Curvature a₁(K)"); ax1.grid(alpha=0.3)
    ax2.plot(Ks, mu_p, 'r-', lw=2); ax2.axhline(0.5, color='gray', ls='--', alpha=0.5)
    ax2.set_xlabel("K"); ax2.set_ylabel("μ"); ax2.set_title("Vertex μ(K)"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "backprop_params_vs_K.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {path}")

    # ── 2. Winner map + iso-C ──
    winner = np.argmax(mean, axis=2)
    img = np.ones((nK, nA, 3))
    for i in range(nK):
        for j in range(nA):
            img[i, j] = to_rgb(METHOD_COLORS[METHODS[winner[i, j]]])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img, aspect='auto', origin='lower', extent=[-0.5, nA-0.5, -0.5, nK-0.5])
    # Draw explicit parabolas K = c(C)·(α-μ(C))² + C
    alpha_fine = np.linspace(0, 1, 200)
    for C_level in range(1, 21):
        c_val, mu_val = model.get_params_at_K0(C_level)
        K_curve = c_val * (alpha_fine - mu_val)**2 + C_level
        x_plot = alpha_fine / 0.05
        y_plot = K_curve - 1  # convert K to index
        mask = (y_plot >= -0.5) & (y_plot <= nK - 0.5)
        if mask.sum() > 1:
            ax.plot(x_plot[mask], y_plot[mask], 'w-', linewidth=0.8, alpha=0.6)
    ax.set_xlim(-0.5, nA-0.5); ax.set_ylim(-0.5, nK-0.5)
    ax.set_xticks(range(0, nA, 2))
    ax.set_xticklabels([f"{A_ARR[i]:.2f}" for i in range(0, nA, 2)], rotation=45, ha='right')
    ax.set_yticks(range(0, nK, 2))
    ax.set_yticklabels([str(int(K_ARR[i])) for i in range(0, nK, 2)])
    ax.set_xlabel("α"); ax.set_ylabel("K")
    ax.set_title("Iso-C contours on winner map", fontsize=13)
    handles = [Patch(facecolor=METHOD_COLORS[m], label=METHOD_LABELS[m]) for m in METHODS]
    ax.legend(handles=handles, loc='upper left', fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "backprop_winner_map.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {path}")

    # ── 3. Rank vs K₀ (Gaussian-weighted along parabolas) ──
    # Same computation as the loss: for each K₀, weight cells by proximity
    K0_values = np.linspace(1, 20, 100)
    eps_plot = 0.5

    # Load full 8-method data for this plot
    d_full = np.load("results/direct_results.npz")
    mean_full = d_full['mean']
    all_m_list = list(d_full['methods'])
    ranks_full = np.zeros_like(mean_full)
    for i in range(nK):
        for j in range(nA):
            order = np.argsort(-mean_full[i, j, :])
            for r, idx in enumerate(order):
                ranks_full[i, j, idx] = r + 1

    ALL_METHODS = ["approval", "borda", "irv", "minimax", "plurality",
                   "random_dictator", "star", "total_score"]
    ALL_LABELS = {"approval": "Approval", "borda": "Borda", "irv": "IRV",
                  "minimax": "Minimax", "plurality": "Plurality",
                  "random_dictator": "Rand. Dict.", "star": "STAR",
                  "total_score": "Total Score"}
    ALL_COLORS = {"approval": "#e41a1c", "borda": "#377eb8", "irv": "#4daf4a",
                  "minimax": "#984ea3", "plurality": "#ff7f00",
                  "random_dictator": "#a65628", "star": "#f781bf", "total_score": "#999999"}

    rank_curves = np.zeros((len(K0_values), 8))
    for ki, K0 in enumerate(K0_values):
        c_val, mu_val = model.get_params_at_K0(K0)
        total_w = 0.0
        w_rank = np.zeros(8)
        for j, alpha in enumerate(A_ARR):
            K_on_curve = c_val * (alpha - mu_val)**2 + K0
            for i, K_cell in enumerate(K_ARR):
                dist = abs(K_cell - K_on_curve)
                w = np.exp(-dist**2 / (2 * eps_plot**2))
                w_rank += w * ranks_full[i, j, :]
                total_w += w
        if total_w > 0:
            rank_curves[ki] = w_rank / total_w

    fig, ax = plt.subplots(figsize=(14, 7))
    complexity = K0_values / 20.0
    for method in ALL_METHODS:
        midx = all_m_list.index(method)
        ax.plot(complexity, rank_curves[:, midx], color=ALL_COLORS[method],
                linewidth=2.5, label=ALL_LABELS[method])
    ax.set_xlabel("Complexity (K₀ / 20)", fontsize=13)
    ax.set_ylabel("Mean rank (1=best, 8=worst)", fontsize=13)
    ax.set_title("Voting method rank vs complexity\n"
                 "(Gaussian-weighted along fitted parabolas)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, ncol=2, loc='center right')
    ax.grid(alpha=0.3); ax.invert_yaxis(); ax.set_yticks(range(1, 9))
    fig.tight_layout()
    path = os.path.join(FIGDIR, "backprop_rank_vs_K0.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {path}")

    # ── 4. Rank heatmaps + parabolas for ALL 8 methods, curves at every K ──
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    cmap = plt.cm.RdYlGn_r
    for ax, method in zip(axes.ravel(), ALL_METHODS):
        midx = all_m_list.index(method)
        im = ax.imshow(ranks_full[:, :, midx], aspect='auto', origin='lower',
                       cmap=cmap, vmin=1, vmax=8, extent=[-0.5, nA-0.5, -0.5, nK-0.5])
        for C_level in range(1, 21):
            c_val, mu_val = model.get_params_at_K0(C_level)
            K_curve = c_val * (alpha_fine - mu_val)**2 + C_level
            x_p = alpha_fine / 0.05
            y_p = K_curve - 1
            m = (y_p >= -0.5) & (y_p <= nK - 0.5)
            if m.sum() > 1:
                ax.plot(x_p[m], y_p[m], 'k-', linewidth=0.5, alpha=0.5)
        ax.set_title(ALL_LABELS[method], fontsize=12, fontweight='bold')
        ax.set_xticks(range(0, nA, 4))
        ax.set_xticklabels([f"{A_ARR[i]:.2f}" for i in range(0, nA, 4)],
                           rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(0, nK, 2))
        ax.set_yticklabels([str(int(K_ARR[i])) for i in range(0, nK, 2)], fontsize=8)
        ax.set_xlabel("α", fontsize=9); ax.set_ylabel("K", fontsize=9)
        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.set_ticks([1,2,3,4,5,6,7,8])
        cb.set_ticklabels(["1st","2nd","3rd","4th","5th","6th","7th","8th"])
        cb.ax.tick_params(labelsize=6)
    fig.suptitle("Rank heatmaps with iso-complexity catenary curves (all K)", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(FIGDIR, "backprop_rank_heatmaps.pdf")
    fig.savefig(path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {path}")


def main():
    print("=== C = K - a₁(K)α² - a₂(K)α ===\n")
    mean = load_data()
    ranks = compute_ranks(mean)
    model, loss = fit(ranks, eps=0.5, lr=0.02, steps=3000)
    print(f"\nFinal loss: {loss:.4f}")
    plot_results(mean, ranks, model)
    print("\nDone.")


if __name__ == "__main__":
    main()
