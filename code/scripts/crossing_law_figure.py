"""Plot cumulative prediction costs for fixed depth-one codes.

This is a theoretical iid-row calculation at d=256, L=L_max=1.
The left panel uses alpha=1/2; the right panel varies alpha from .1 to .9.
It substitutes mean visits into the iid-row profiles. It neither measures
a Markov-chain crossing nor verifies the chain convergence hypotheses.

    python code/scripts/crossing_law_figure.py \
        --out figures/crossing_law.pdf \
        --json results/mechanism/crossing_law_example.json

Dependencies: NumPy, SciPy, Matplotlib. All information quantities are bits.
Use --smoke for a cheaper quadrature and fewer plotted points.

With c=1-alpha and t(u)=c*u**(-alpha), the profiles are
Gamma(nu)=a1*nu/ln(2)-integral r1(nu*t(u)*t(v)) du dv,
Pi1(nu)=integral g(nu*t(u)) du-integral g(nu*t(u)*t(v)) du dv,
where g is geometric entropy. The second formula also equals the
integrated KL divergence Geom(s*t(v)) || Geom(s), s=nu*t(u).
For one normalized iid row it follows from the exact mismatch
log2(binomial(k+d-1,d-1))-H(counts), followed by the count-entropy limit.

The substitution z=-ln(u*v) changes a double integral of f(u*v) into
integral_0^infinity z*exp(-z)*f(exp(-z)) dz. We use Gauss-Laguerre
quadrature and independently check the outer integration adaptively.

The geometric sum in master_curve.r1_geometric is an independent check
on this stable, exact integral (shown here in nats):

 r1(x)*ln(2) = integral_0^1 x/(1+x*y) *
                [1/(-ln(1-y))-(1-y)/y] dy.

It follows by summing E ln(N!)=sum_{j>=1}(x/(1+x))**j ln(j) under
the geometric distribution, using the integral for ln(j), and the
integral gamma=integral_0^1 [1/y-1/(-ln(1-y))]dy.
To improve quadrature near zero we subtract the first two coefficients
of C(y)=[1/(-ln(1-y))-1/y+1/2]/y, namely -1/12-y/24, and integrate
these terms analytically. No large-x extrapolation of r1 is needed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit, roots_laguerre
from matplotlib.colors import TwoSlopeNorm

from master_curve import EG, L2, r1_geometric


ALPHA = 0.5
D = 256
A1 = 1.0 - EG


class Profiles:
    """Stable scalar curves and integrated row profiles."""

    def __init__(self, order: int, alpha: float = ALPHA):
        self.order = order
        self.alpha = alpha
        self.c = 1.0 - alpha
        self.z, self.w = roots_laguerre(order)
        y = -np.expm1(-self.z)
        c_y = (1.0 / self.z + 0.5 - 1.0 / y) / y
        self.log_y = np.log(y)
        self.remainder_weights = self.w * (c_y + 1.0 / 12.0 + y / 24.0)

    def r1_log(self, log_x):
        """Exact depth-one information integral, evaluated at log(x)."""
        log_x = np.asarray(log_x, dtype=float)
        log_one_plus_x = np.logaddexp(0.0, log_x)
        small = log_x < math.log(0.1)
        # The clipped branch is unused for small x; clipping avoids overflow.
        inv_x = np.exp(-np.maximum(log_x, math.log(0.1)))
        j0 = 1.0 - inv_x * log_one_plus_x
        j1 = 0.5 - inv_x * j0
        if np.any(small):
            x_small = np.exp(np.minimum(log_x, math.log(0.1)))
            s0 = np.zeros_like(log_x)
            s1 = np.zeros_like(log_x)
            power = x_small.copy()
            for k in range(1, 20):
                s0 += power / (k + 1)
                s1 += power / (k + 2)
                power *= -x_small
            j0 = np.where(small, s0, j0)
            j1 = np.where(small, s1, j1)
        correction = expit(log_x[..., None] + self.log_y) @ self.remainder_weights
        return (0.5 * log_one_plus_x - j0 / 12.0 - j1 / 24.0 + correction) / L2

    @staticmethod
    def geometric_entropy_log(log_x):
        """Geometric entropy in bits, stable also for enormous means."""
        log_x = np.asarray(log_x, dtype=float)
        # x*ln(1+1/x)=1+O(1/x); the omitted term is <5e-14 here.
        second = np.where(
            log_x > 30.0,
            1.0,
            np.exp(np.minimum(log_x, 30.0)) * np.logaddexp(0.0, -log_x),
        )
        return (np.logaddexp(0.0, log_x) + second) / L2

    def evaluate(self, nu: float) -> tuple[float, float]:
        if nu == 0.0:
            return 0.0, 0.0
        if nu < 0.0:
            raise ValueError("nu must be nonnegative")
        base = math.log(nu) + self.alpha * self.z
        pair_log_x = base + 2.0 * math.log(self.c)
        gamma = A1 * nu / L2 - (self.w * self.z) @ self.r1_log(pair_log_x)
        if self.alpha == 0.0:
            pi = 0.0
        else:
            single = self.w @ self.geometric_entropy_log(base + math.log(self.c))
            pair = (self.w * self.z) @ self.geometric_entropy_log(pair_log_x)
            pi = single - pair
        return float(gamma), float(pi)

    @property
    def ceiling(self) -> float:
        return (-self.alpha - math.log1p(-self.alpha)) / L2

    def roots(self) -> tuple[float, float]:
        if self.alpha == 0.0:
            return 0.0, 0.0
        root = brentq(lambda nu: np.subtract(*self.evaluate(nu)), 1e-5, 4.0,
                      xtol=2e-13, rtol=1e-13)
        ceiling_root = brentq(lambda nu: self.evaluate(nu)[0] - self.ceiling,
                              1e-5, 4.0, xtol=2e-13, rtol=1e-13)
        return root, ceiling_root

    def adaptive(self, nu: float) -> tuple[float, float]:
        """Independent outer quadrature; the exponential tail after 80 is tiny."""
        log_nu = math.log(nu)
        log_c = math.log(self.c)

        def integrate(fun):
            return quad(fun, 0.0, 80.0, epsabs=2e-12, epsrel=2e-12, limit=200)[0]

        row_r = integrate(lambda z: z * math.exp(-z) * float(
            self.r1_log(log_nu + 2 * log_c + self.alpha * z)))
        single_h = integrate(lambda z: math.exp(-z) * float(
            self.geometric_entropy_log(log_nu + log_c + self.alpha * z)))
        pair_h = integrate(lambda z: z * math.exp(-z) * float(
            self.geometric_entropy_log(log_nu + 2 * log_c + self.alpha * z)))
        return A1 * nu / L2 - row_r, single_h - pair_h


def checks(profile: Profiles, smoke: bool) -> dict:
    scalar = []
    for x in (0.001, 0.1, 1.0, 8.0, 61.0, 300.0, 1000.0):
        integral = float(profile.r1_log(math.log(x)))
        reference = float(r1_geometric(x))
        scalar.append(dict(x=x, integral_bits=integral, geometric_bits=reference,
                           difference_bits=integral - reference))
    scalar_max = max(abs(v["difference_bits"]) for v in scalar)
    assert scalar_max < (2e-7 if smoke else 2e-9), scalar_max

    refinements = []
    levels = (16, 32, 64) if smoke else (64, 128, 256)
    comparison_nus = np.geomspace(0.01, 2.0, 41)
    previous = None
    for order in levels:
        current = Profiles(order)
        roots = current.roots()
        values = np.array([current.evaluate(nu) for nu in comparison_nus])
        row = dict(order=order, nu_row=roots[0], nu_ceiling=roots[1])
        if previous is not None:
            row.update(
                max_profile_change_bits=float(np.max(np.abs(values - previous[1]))),
                root_change=abs(roots[0] - previous[0][0]),
                ceiling_root_change=abs(roots[1] - previous[0][1]),
            )
        refinements.append(row)
        previous = roots, values
    assert refinements[-1]["max_profile_change_bits"] < (2e-5 if smoke else 1e-8)

    adaptive = []
    for nu in (0.05, profile.roots()[0], 1.0, profile.roots()[1]):
        laguerre = profile.evaluate(nu)
        independent = profile.adaptive(nu)
        differences = np.subtract(laguerre, independent)
        adaptive.append(dict(nu=nu, laguerre_bits=list(laguerre),
                             adaptive_bits=list(independent),
                             difference_bits=differences.tolist()))
    adaptive_max = max(max(abs(x) for x in row["difference_bits"]) for row in adaptive)
    assert adaptive_max < (2e-7 if smoke else 2e-10), adaptive_max

    uniform = Profiles(profile.order, alpha=0.0)
    uniform_checks = []
    for nu in (0.001, 0.1, 1.0, 8.0):
        gamma, pi = uniform.evaluate(nu)
        reference = A1 * nu / L2 - r1_geometric(nu)
        # Compute the unsimplified difference as well, so setting Pi=0 does
        # not make this test merely repeat the implementation's special case.
        entropy = float(uniform.geometric_entropy_log(math.log(nu)))
        raw_pi = (uniform.w.sum() - uniform.w @ uniform.z) * entropy
        uniform_checks.append(dict(nu=nu, gamma_bits=gamma, phi1_bits=reference,
                                   gamma_difference_bits=gamma - reference,
                                   pi_bits=pi, unsimplified_pi_bits=float(raw_pi)))
    assert uniform.ceiling == 0.0
    assert max(abs(v["gamma_difference_bits"]) for v in uniform_checks) < (
        2e-7 if smoke else 2e-10)
    assert max(abs(v["unsimplified_pi_bits"]) for v in uniform_checks) < 2e-12
    _, pi_tail = profile.evaluate(1e8)
    assert abs(pi_tail - profile.ceiling) < 1e-7
    return dict(
        scalar_geometric_crosscheck=scalar,
        scalar_max_absolute_difference_bits=scalar_max,
        quadrature_refinement=refinements,
        refinement_nus=comparison_nus.tolist(),
        independent_adaptive_outer_quadrature=adaptive,
        adaptive_max_absolute_difference_bits=adaptive_max,
        alpha_zero_checks=uniform_checks,
        alpha_zero_ceiling=uniform.ceiling,
        ceiling_limit_check=dict(nu=1e8, pi_bits=pi_tail,
                                ceiling_bits=profile.ceiling),
        interpretation="These are numerical checks at stated points, not a rigorous error certificate.",
    )


def zipf_entropy(alpha: float, d: int = D) -> float:
    q = np.arange(1, d + 1, dtype=float) ** (-alpha)
    q /= q.sum()
    return float(-q @ np.log2(q))


def cost_data(profile: Profiles, smoke: bool) -> dict:
    root, _ = profile.roots()
    crossing = root * D**2
    n = np.unique(np.r_[np.geomspace(100, 3e6, 81 if smoke else 201), crossing])
    nu = n / D**2
    values = np.array([profile.evaluate(float(x)) for x in nu])
    gap = (values[:, 0] - values[:, 1]) / nu
    hq = zipf_entropy(ALPHA)
    alphas = np.linspace(0.1, 0.9, 11 if smoke else 41)
    map_n = np.geomspace(100, 3e6, 51 if smoke else 101)
    map_nu = map_n / D**2
    map_gamma = np.empty((len(map_n), len(alphas)))
    map_pi = np.empty_like(map_gamma)
    map_hq = np.array([zipf_entropy(float(alpha)) for alpha in alphas])
    boundaries = []
    for j, alpha in enumerate(alphas):
        current = Profiles(profile.order, float(alpha))
        pair = np.array([current.evaluate(float(x)) for x in map_nu])
        map_gamma[:, j], map_pi[:, j] = pair[:, 0], pair[:, 1]
        root_j = brentq(lambda x: np.subtract(*current.evaluate(x)),
                        map_nu[0], map_nu[-1], xtol=2e-12)
        boundaries.append(float(root_j * D**2))
    map_gap = (map_gamma - map_pi) / map_nu[:, None]
    map_frequency = np.broadcast_to(map_hq, map_gap.shape).copy()
    map_copying = map_frequency - map_gap
    return dict(
        d=D, source_depth=1, code_depth_cap=1,
        convention="Cumulative expected codelength divided by n; not instantaneous loss.",
        frequency_formula="ell_F(n)/n = H(q_d), using the exact finite-d Zipf entropy",
        copying_formula="ell_C(n)/n = H(q_d) - [Gamma(n/d^2)-Pi1(n/d^2)]/(n/d^2)",
        omitted_terms="Finite-n frequency-code remainder and the difference between the mean-visit profiles and the actual chain.",
        left=dict(alpha=ALPHA, zipf_entropy_bits=hq, n=n.tolist(), nu=nu.tolist(),
                  gamma_bits=values[:, 0].tolist(), pi_bits=values[:, 1].tolist(),
                  frequency_bits_per_token=np.full_like(n, hq).tolist(),
                  copying_bits_per_token=(hq - gap).tolist(),
                  frequency_minus_copying_bits_per_token=gap.tolist(),
                  crossing_n=crossing),
        heatmap=dict(alpha=alphas.tolist(), n=map_n.tolist(), nu=map_nu.tolist(),
                     zipf_entropy_bits=map_hq.tolist(), matrix_axes=["sequence_length", "alpha"],
                     gamma_bits=map_gamma.tolist(), pi_bits=map_pi.tolist(),
                     frequency_bits_per_token=map_frequency.tolist(),
                     copying_bits_per_token=map_copying.tolist(),
                     frequency_minus_copying_bits_per_token=map_gap.tolist(),
                     crossing_n=boundaries),
    )


def check_heatmap(order: int, smoke: bool) -> list[dict]:
    """Check representative exponents without a separate tail experiment."""
    lengths = np.array([100, 1000, 10000, 100000, 1000000, 3000000], dtype=float)
    nus = lengths / D**2
    checked = []
    for alpha in (0.2, 0.5, 0.8):
        coarse, fine = Profiles(order // 2, alpha), Profiles(order, alpha)
        a = np.array([coarse.evaluate(float(x)) for x in nus])
        b = np.array([fine.evaluate(float(x)) for x in nus])
        gap_difference = ((a[:, 0] - a[:, 1]) - (b[:, 0] - b[:, 1])) / nus
        error = float(np.max(np.abs(gap_difference)))
        assert error < (1e-5 if smoke else 1e-7), (alpha, error)
        checked.append(dict(alpha=alpha, n=lengths.tolist(),
                            coarse_order=order // 2, fine_order=order,
                            max_profile_change_bits=float(np.max(np.abs(a - b))),
                            gap_changes_bits_per_token=gap_difference.tolist(),
                            max_gap_change_bits_per_token=error))
    return checked


def draw(out: Path, data: dict):
    plt.rcParams.update({
        "font.size": 8, "mathtext.fontset": "cm", "font.family": "serif",
        "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.65, "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "pdf.fonttype": 42,
    })
    blue, orange, gray = "#27618A", "#BD5E2C", "#626262"
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.65), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.17]})
    left, right = axes
    costs = data["left"]
    n = np.asarray(costs["n"])
    frequency = np.asarray(costs["frequency_bits_per_token"])
    copying = np.asarray(costs["copying_bits_per_token"])
    crossing, hq = costs["crossing_n"], costs["zipf_entropy_bits"]
    left.plot(n, frequency, color=orange, lw=1.5, label="Frequency prediction")
    left.plot(n, copying, color=blue, lw=1.6, label="Copying")
    left.axvline(crossing, color=gray, lw=0.7, ls=":")
    left.plot(crossing, hq, "o", color=gray, ms=3)
    left.annotate(f"Break-even\n{crossing:,.0f} tokens", xy=(crossing, hq),
                  xytext=(1.1e5, hq + 0.18), fontsize=7.5,
                  arrowprops=dict(arrowstyle="-", color=gray, lw=0.65))
    left.set(xscale="log", xlim=(n[0], n[-1]), ylim=(hq - 0.63, hq + 0.49),
             xlabel="Sequence length $n$", ylabel="Expected codelength / $n$ (bits)")
    left.set_title(r"$\alpha=0.5$, $d=256$", fontsize=9)
    left.legend(loc="lower left", fontsize=6.6, frameon=False, handlelength=1.7)
    left.grid(True, axis="y", color="#dddddd", lw=0.45)
    heat = data["heatmap"]
    alphas, lengths = np.asarray(heat["alpha"]), np.asarray(heat["n"])
    gap = np.asarray(heat["frequency_minus_copying_bits_per_token"])
    norm = TwoSlopeNorm(vmin=-2.4, vcenter=0.0, vmax=0.6)
    mesh = right.pcolormesh(alphas, lengths, gap, shading="nearest",
                           cmap="RdBu", norm=norm, rasterized=True)
    right.contour(alphas, lengths, gap, levels=[0.0], colors=["#333333"], linewidths=0.8)
    right.text(0.50, 6e5, "Copying cheaper", ha="center", va="center",
               fontsize=7.5, color="white")
    right.text(0.54, 400, "Frequency\nprediction cheaper", ha="center", va="center",
               fontsize=7.5, color="#222222")
    right.set(yscale="log", xlim=(0.1, 0.9), ylim=(lengths[0], lengths[-1]),
              xlabel=r"Zipf exponent $\alpha$", ylabel="Sequence length $n$")
    right.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    right.set_title("$d=256$", fontsize=9)
    colorbar = fig.colorbar(mesh, ax=right, fraction=0.065, pad=0.025,
                           ticks=[-2.0, -1.0, 0.0, 0.5])
    colorbar.set_label("Frequency minus copying (bits/token)", fontsize=7.1)
    colorbar.ax.tick_params(labelsize=6.8, width=0.5, length=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"Title": "Cumulative prediction costs"})
    fig.savefig(out.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures/crossing_law.pdf"))
    parser.add_argument("--json", type=Path,
                        default=Path("results/mechanism/crossing_law_example.json"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    profile = Profiles(64 if args.smoke else 256)
    validation = checks(profile, args.smoke)
    validation["heatmap_alpha_refinement"] = check_heatmap(profile.order, args.smoke)
    plotted = cost_data(profile, args.smoke)
    root, upper = profile.roots()
    nus = np.unique(np.r_[np.linspace(0.0, 1.5, 81 if args.smoke else 301), root, upper])
    curves = np.array([profile.evaluate(float(nu)) for nu in nus])
    assert np.all(curves[:, 0] >= -1e-12)
    assert np.all(curves[:, 1] >= -1e-12)
    assert np.all(curves[:, 1] <= profile.ceiling + 1e-12)
    assert np.all(np.diff(curves[:, 0]) > 0)
    assert np.all(np.diff(curves[:, 1]) > 0)
    gap = curves[:, 0] - curves[:, 1]
    assert np.all(gap[(nus > 0) & (nus < root - 1e-9)] < 0)
    assert np.all(gap[nus > root + 1e-9] > 0)
    output = dict(
        schema_version=2,
        parameters=dict(alpha=ALPHA, source_depth=1, code_depth_cap=1),
        interpretation=("Theoretical fixed-depth-one iid-row approximation with mean visits substituted. "
                        "The roots are not measured chain crossings and do not verify the chain hypotheses. "
                        "The default larger depth mixture is a different code."),
        formulas=dict(
            t="t(u)=(1-alpha)*u**(-alpha)",
            gamma="Gamma(nu)=a1*nu/ln(2)-integral_0^1 integral_0^1 r1(nu*t(u)*t(v)) du dv",
            pi="Pi1(nu)=integral_0^1 g(nu*t(u))du-integral_0^1 integral_0^1 g(nu*t(u)*t(v))du dv",
            g="g(x)=((1+x)*ln(1+x)-x*ln(x))/ln(2)",
            ceiling="u(alpha)=(-alpha-ln(1-alpha))/ln(2)",
        ),
        quadrature_order=profile.order,
        nu_row=root,
        nu_ceiling=upper,
        ceiling_bits=profile.ceiling,
        crossing_profile_bits=profile.evaluate(root)[0],
        nu=nus.tolist(), gamma_bits=curves[:, 0].tolist(), pi_bits=curves[:, 1].tolist(),
        plotted_costs=plotted,
        validation=validation,
        reproduction=("python code/scripts/crossing_law_figure.py --out figures/crossing_law.pdf "
                      "--json results/mechanism/crossing_law_example.json"),
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    draw(args.out, plotted)
    print(json.dumps({k: output[k] for k in ("nu_row", "nu_ceiling", "ceiling_bits")}, indent=2))
    print("Checks passed; wrote", args.out, args.out.with_suffix(".png"), "and", args.json)


if __name__ == "__main__":
    main()
