"""Write the manuscript tables for the analytic unigram calculation."""
from __future__ import annotations

import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n")
    temporary.replace(path)


def exact_table(data: dict) -> str:
    rows = []
    for cell in data["cases"]:
        exact = cell["markov_types"]["expected_cost_bits"]
        approximation = cell["profile_curvature_approximation"]
        jensen = approximation["jensen_bits"] - exact
        corrected = approximation["corrected_bits"] - exact
        fast = cell["fast_minus_exact_bits"]
        rows.append(
            f'{cell["d"]} & {cell["alpha"]:g} & {cell["source_depth"]} & '
            f'{cell["n"]} & {cell["coding_cap"]} & {exact:.6f} & '
            f'{jensen:+.6f} & {corrected:+.6f} & {fast:+.6f} \\\\')
    return r"""\begin{table}[!htbp]
\centering
\scriptsize
\begin{tabular}{rrrrrrrrr}
\toprule
$d$ & $\alpha$ & $L_{\mathrm{s}}$ & $k$ & $L_{\max}$ & exact bits
 & \multicolumn{3}{c}{error (bits)} \\
\cmidrule(lr){7-9}
 & & & & & & mean profile & second order & scalable \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Sampling-free small-case checks. ``Exact bits'' evaluates
the exact sum in \eqref{eq:exact-unigram-cost}. The last three columns
subtract this value from
$[\ln\Gamma(k)-A(\nu)]/\ln2$, the second-order expression in
\eqref{eq:unigram-profile-approximation}, and the scalable
formula \eqref{eq:scalable-unigram-cost}, respectively.}
\label{tab:unigram-exact-checks}
\end{table}
"""


def group_row(label: str, cells: list[dict]) -> str:
    relative = [100 * cell["absolute_relative_error"] for cell in cells]
    standardized = [abs(cell["difference_in_measured_SEs"]) for cell in cells]
    return (f'{label} & {len(cells)} & {statistics.median(relative):.3f} & '
            f'{max(relative):.3f} & {sum(value <= 2 for value in standardized)}/'
            f'{len(cells)} \\\\')


def large_table(data: dict) -> str:
    main = data["rows"][:18]
    rows = [
        group_row(r"$L_{\mathrm{s}}=2$", [x for x in main if x["source_depth"] == 2]),
        group_row(r"$L_{\mathrm{s}}=3$", [x for x in main if x["source_depth"] == 3]),
        group_row("all settings", main),
    ]
    return r"""\begin{table}[!htbp]
\centering
\small
\begin{tabular}{lrrrc}
\toprule
source depths & parameter settings & median $|\mathrm{error}|$ (\%)
 & maximum $|\mathrm{error}|$ (\%) & within $2$ SE \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{A one-checkpoint comparison for each of the 18 parameter settings.  Earlier
simulations recorded ten lengths around an older crossing estimate
$n_{\mathrm{old}}$.  This table uses the recorded length equal to
$n_{\mathrm{old}}$, up to rounding.  The older estimate selects the test
length but is not an input to the unigram formula.  ``Within $2$ SE''
counts settings where the formula differs from the simulation mean by
no more than twice that mean's sample standard error.  Each mean uses
16 saved source tables and paths; the formula itself uses no samples.}
\label{tab:unigram-large-checks}
\end{table}
"""


def main() -> None:
    exact_path = ROOT / "results/mechanism/unigram_annealed_exact_validation.json"
    large_path = ROOT / "results/mechanism/unigram_analytic_validation.json"
    exact = json.loads(exact_path.read_text())
    large = json.loads(large_path.read_text())
    write(ROOT / "results/mechanism/unigram_analytic_exact_table.tex",
          exact_table(exact))
    write(ROOT / "results/mechanism/unigram_analytic_large_table.tex",
          large_table(large))


if __name__ == "__main__":
    main()
