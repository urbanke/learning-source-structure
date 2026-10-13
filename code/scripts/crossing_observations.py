"""Describe the saved crossing experiments, without fitting a scaling law.

Collect (requires the original ignored output/ files):
    python code/scripts/crossing_observations.py --collect
Verify the archive against those originals:
    python code/scripts/crossing_observations.py --check
Render the atlas from the small committed archive only:
    python code/scripts/crossing_observations.py --plot
Make the editable rich-text version of the reviewed findings:
    python code/scripts/crossing_observations.py --report

Only configurations with a located crossing for EVERY run enter the main
median curves. A positive first check is recorded separately; it is not
silently treated as a last crossing. All observations are finite-horizon.
The saved 'exact' search includes some non-unit-width brackets.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO / "results/mechanism/crossing_observations.json"
EXACT_NAME = re.compile(r"exact_d(\d+)_a([\d.]+)_L(\d+)(?:_s(\d+))?\.json")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def quantile(values, p):
    """Linear interpolation, the usual numpy default, with no dependency."""
    values = sorted(values)
    if not values:
        return None
    h = (len(values) - 1) * p
    i = math.floor(h)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (h - i)


def describe(values):
    if not values:
        return None
    assert all(math.isfinite(v) and v > 0 for v in values)
    mean = statistics.mean(values)
    return dict(n=len(values), mean=mean, median=statistics.median(values),
                q25=quantile(values, .25), q75=quantile(values, .75),
                q10=quantile(values, .1), q90=quantile(values, .9),
                minimum=min(values), maximum=max(values),
                sd=statistics.stdev(values) if len(values) > 1 else None)


def read(path):
    data = path.read_bytes()
    return json.loads(data), (path.relative_to(REPO).as_posix(), sha(data))


def file_digest(items):
    return sha("".join(f"{name}\t{digest}\n" for name, digest in sorted(items)).encode())


def collect_exact():
    groups = defaultdict(list)
    provenance = defaultdict(list)
    all_sources = []
    for path in sorted((REPO / "output/exact").glob("exact_*.json")):
        match = EXACT_NAME.fullmatch(path.name)
        if not match:
            continue
        data, source = read(path)
        d, alpha, depth, file_seed = match.groups()
        key = int(d), float(alpha), int(depth)
        assert key == tuple(data[k] for k in ("d", "alpha", "L")), path
        if file_seed is not None:
            assert len(data["seeds"]) == 1 and data["seeds"][0]["seed"] == int(file_seed)
        groups[key].extend((row, data["n_cap"], data.get("mode")) for row in data["seeds"])
        provenance[key].append(source)
        all_sources.append(source)
    records = []
    for (d, alpha, depth), rows in sorted(groups.items()):
        seeds = [row["seed"] for row, _, _ in rows]
        assert len(set(seeds)) == len(seeds), (d, alpha, depth, "duplicate seed")
        assert sorted(seeds) == list(range(len(seeds))), (d, alpha, depth, "seed gap")
        located, widths, endpoints, positive_checks = [], [], [], []
        censored = 0
        for row, _, _ in rows:
            value = row.get("first_pos")
            note = row.get("note", "")
            if note.startswith("censored") or value is None:
                censored += 1
                continue
            assert isinstance(value, int) and value > 0
            endpoints.append(value)
            if note == "positive from the start":
                positive_checks.append(value)
                continue
            lower = row.get("last_neg")
            assert isinstance(lower, int) and 0 < lower < value, row
            located.append(value)
            widths.append(value - lower)
        sources = provenance[d, alpha, depth]
        records.append(dict(
            instrument="refined", d=d, alpha=alpha, L=depth, n_seeds=len(seeds),
            n_located=len(located), n_positive_first_check=len(positive_checks),
            positive_first_check_values=sorted(set(positive_checks)), n_censored=censored,
            located=describe(located), recorded_endpoints=describe(endpoints),
            all_runs_located=len(located) == len(seeds),
            bracket_width_max=max(widths) if widths else None,
            n_unit_brackets=sum(w == 1 for w in widths),
            bracket_relative_width_max=max((w/v for w, v in zip(widths, located)), default=None),
            n_cap_min=min(cap for _, cap, _ in rows), n_cap_max=max(cap for _, cap, _ in rows),
            search_modes=sorted({mode for _, _, mode in rows if mode}),
            source_file_count=len(sources), source_group_sha256=file_digest(sources)))
    assert records, "No refined-search sources found"
    return records, all_sources


def mean_crossing(paths):
    common = sorted(set.intersection(*(set(p["checkpoints"]) for p in paths)))
    if not common:
        return None, dict(status="no_common_checkpoints", common_horizon=None)
    maps = [dict(zip(p["checkpoints"], p["lam"])) for p in paths]
    means = [sum(m[t] for m in maps) / len(maps) for t in common]
    crossing = None
    upcrossings = 0
    for i in range(1, len(common)):
        if means[i - 1] <= 0 < means[i]:
            w = -means[i - 1] / (means[i] - means[i - 1])
            crossing = math.exp(math.log(common[i - 1]) + w * math.log(common[i] / common[i - 1]))
            upcrossings += 1
    if means[-1] <= 0:
        crossing = None
        status = "nonpositive_at_common_end"
    elif crossing is None:
        assert min(means) > 0
        status = "positive_at_every_common_check"
    else:
        status = "located"
    return crossing, dict(status=status, common_horizon=common[-1],
                          common_start=common[0], common_points=len(common),
                          mean_at_common_end=means[-1], mean_minimum=min(means),
                          observed_upcrossings=upcrossings,
                          min_path_horizon=min(p["horizon"] for p in paths),
                          max_path_horizon=max(p["horizon"] for p in paths))


def collect_checkpoint():
    records, all_sources = [], []
    keys = set()
    for path in sorted((REPO / "output/takeover").glob("*/summary_*.json")):
        data, source = read(path)
        all_sources.append(source)
        key = tuple(data[k] for k in ("d", "alpha", "L"))
        assert key not in keys, key
        keys.add(key)
        assert len(set(data["seeds"])) == len(data["seeds"])
        stem = path.stem.removeprefix("summary_")
        paths, raw_sources = [], []
        for seed in data["seeds"]:
            raw, raw_source = read(path.parent / f"{stem}_seed{seed}.json")
            assert tuple(raw[k] for k in ("d", "alpha", "L")) == key
            assert raw["seed"] == seed
            assert len(raw["checkpoints"]) == len(raw["lam"]) == len(raw["l_F"]) == len(raw["l_C"])
            assert raw["checkpoints"] == sorted(set(raw["checkpoints"]))
            assert all(math.isfinite(v) for k in ("lam", "l_F", "l_C") for v in raw[k])
            assert all(math.isclose(f-c, gap, rel_tol=1e-9, abs_tol=1e-6)
                       for f, c, gap in zip(raw["l_F"], raw["l_C"], raw["lam"]))
            paths.append(raw)
            raw_sources.append(raw_source)
        all_sources.extend(raw_sources)
        n_censored = sum(bool(p["censored"]) for p in paths)
        assert n_censored == data["n_censored"]
        endpoints = sorted(p["n_star_last"] for p in paths if not p["censored"])
        assert endpoints == data["n_star_last"]
        located = [p["n_star_last"] for p in paths if not p["censored"] and any(v <= 0 for v in p["lam"])]
        always_positive = sum(all(v > 0 for v in p["lam"]) for p in paths)
        mean, diagnostics = mean_crossing(paths)
        saved_mean = data["mean_lambda_crossing"]
        assert (mean is None) == (saved_mean is None), key
        if mean is not None:
            assert math.isclose(mean, saved_mean, rel_tol=1e-10), (key, mean, saved_mean)
        # Preserve the exact archived number after independently reproducing it.
        records.append(dict(
            instrument="checkpoint", d=key[0], alpha=key[1], L=key[2],
            n_seeds=len(paths), n_located=len(located),
            n_positive_all_checks=always_positive, n_censored=n_censored,
            located=describe(located), recorded_endpoints=describe(endpoints),
            all_runs_located=len(located) == len(paths),
            mean_curve_crossing=saved_mean, mean_curve_diagnostics=diagnostics,
            coding_cutoffs=sorted({p["l_max"] for p in paths}),
            sequence_estimators=sorted({p.get("sequence_estimator", "unrecorded") for p in paths}),
            n_cap=data["n_cap"], source=source[0], source_sha256=source[1],
            raw_source_count=len(raw_sources), raw_group_sha256=file_digest(raw_sources)))
    assert records, "No checkpoint sources found"
    return sorted(records, key=lambda r: (r["d"], r["alpha"], r["L"])), all_sources


def summaries(exact, checkpoint):
    index = {(r["d"], r["alpha"], r["L"]): r for r in checkpoint}
    overlap = []
    for r in exact:
        other = index.get((r["d"], r["alpha"], r["L"]))
        if r["d"] <= 1024 and other and r["recorded_endpoints"] and r["recorded_endpoints"]["median"] > 32 and other["recorded_endpoints"]:
            overlap.append(other["recorded_endpoints"]["median"] / r["recorded_endpoints"]["median"])
    ratios = [r["mean_curve_crossing"] / r["recorded_endpoints"]["median"]
              for r in checkpoint if r["mean_curve_crossing"] and r["recorded_endpoints"]]
    inventory = []
    for d in sorted({r["d"] for r in exact}):
        rows = [r for r in exact if r["d"] == d]
        inventory.append(dict(d=d, configurations=len(rows),
                              repetition_counts=dict(sorted(Counter(r["n_seeds"] for r in rows).items())),
                              runs=sum(r["n_seeds"] for r in rows),
                              positive_first_check=sum(r["n_positive_first_check"] for r in rows),
                              all_runs_located_configurations=sum(r["all_runs_located"] for r in rows)))
    return dict(refined_configurations=len(exact), refined_runs=sum(r["n_seeds"] for r in exact),
                refined_located_runs=sum(r["n_located"] for r in exact),
                refined_positive_first_check_runs=sum(r["n_positive_first_check"] for r in exact),
                refined_unit_brackets=sum(r["n_unit_brackets"] for r in exact),
                refined_inventory=inventory, checkpoint_configurations=len(checkpoint),
                checkpoint_runs=sum(r["n_seeds"] for r in checkpoint),
                checkpoint_uncensored_configurations=sum(r["n_censored"] == 0 for r in checkpoint),
                mean_curve_status_counts=dict(Counter(r["mean_curve_diagnostics"]["status"] for r in checkpoint)),
                checkpoint_to_refined_recorded_median_ratio_small_d=describe(overlap),
                mean_curve_to_recorded_median_ratio=describe(ratios),
                mean_curve_to_median_outside_factor_1_5=sum(x < 1/1.5 or x > 1.5 for x in ratios),
                mean_curve_to_median_outside_factor_2=sum(x < .5 or x > 2 for x in ratios))


def collect():
    exact, sources1 = collect_exact()
    checkpoint, sources2 = collect_checkpoint()
    return dict(schema_version=1, generator="code/scripts/crossing_observations.py",
                notes=[
                    "No new simulations and no fitted scaling laws.",
                    "Source L generates the continuation table; coding cutoffs are recorded separately.",
                    "Refined data combine aggregate files at d<=1024 with disjoint per-seed files at larger d.",
                    "A refined positive-first-check flag is not a located last crossing and is not assigned one.",
                    "Located medians/IQRs are conditional on a bracket being found; plots require every run to be located.",
                    "recorded_endpoints includes early-positive flags for auditing historical summaries only.",
                    "Refined locations are upper endpoints of observed negative-positive brackets, sometimes wider than one token.",
                    "Checkpoint path locations are checkpoint endpoints, not interpolated individual crossings.",
                    "Mean-curve zeros are interpolated in log length and independently reproduced from raw paths.",
                    "IQR shows between-run variation, not a confidence interval for a median.",
                    "Refined markers are located negative-to-positive brackets; some are followed by observed reversals in paired checkpoint paths.",
                    "No instrument certifies a final crossing for all later lengths. See crossing_search_audit.json for measured search failures.",
                    "Group hash: SHA256 of sorted relative-path<TAB>file-SHA256<NEWLINE> entries.",
                ],
                source_counts=dict(refined=len(sources1), checkpoint=len(sources2)),
                all_sources_sha256=file_digest(sources1 + sources2),
                summary=summaries(exact, checkpoint), refined=exact, checkpoint=checkpoint)


def save(data, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    # One compact record per line keeps the reviewed archive reasonably small.
    meta = {k: v for k, v in data.items() if k not in ("refined", "checkpoint")}
    text = json.dumps(meta, indent=2, allow_nan=False).rstrip()[:-1].rstrip()
    for name in ("refined", "checkpoint"):
        text += ',\n  "' + name + '": [\n'
        text += ',\n'.join('    ' + json.dumps(r, separators=(',', ':'), allow_nan=False) for r in data[name])
        text += '\n  ]'
    out.write_text(text + '\n}\n')
    fields = ["instrument", "d", "alpha", "L", "n_seeds", "n_located", "n_censored", "all_runs_located",
              "located_median", "located_q25", "located_q75", "positive_first_or_all_checks", "mean_curve_crossing"]
    with out.with_suffix(".tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for name in ("refined", "checkpoint"):
            for r in data[name]:
                row = {k: r.get(k) for k in fields[:8]}
                row.update({"located_" + k: (r["located"] or {}).get(k) for k in ("median", "q25", "q75")})
                row["positive_first_or_all_checks"] = r.get("n_positive_first_check", r.get("n_positive_all_checks"))
                row["mean_curve_crossing"] = r.get("mean_curve_crossing")
                writer.writerow({k: "NA" if v is None else v for k, v in row.items()})


def plot(data, out):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "pdf.fonttype": 42})
    alphas = sorted({r["alpha"] for r in data["refined"]})
    depths = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24]
    ds = sorted({r["d"] for r in data["refined"]})
    colors = {"refined": "#176c91", "checkpoint": "#c16a18", "mean": "#2c885f"}
    powers = json.loads((REPO / "results/mechanism/crossing_power_fits.json").read_text())
    assert powers["source_sha256"] == sha((REPO / powers["source"]).read_bytes())
    assert data == json.loads((REPO / powers["source"]).read_text())
    fits = {(r["method"], r["window"], r["alpha"], r["L"]): r
            for r in powers["records"]}

    def footer(fig, text):
        fig.text(.06, .025, text, fontsize=8.5, color="#444", va="bottom")

    def page_number(fig, number):
        fig.text(.975, .022, str(number), ha="right", fontsize=9, color="#444")

    def panel(ax, alpha, depth, mean=False):
        for instrument, marker in (("refined", "o"), ("checkpoint", "s")):
            rows = {(r["d"]): r for r in data[instrument] if r["alpha"] == alpha and r["L"] == depth}
            xs = sorted(rows)
            vals = [rows[d]["located"]["median"] if rows[d]["all_runs_located"] else np.nan for d in xs]
            lower = [rows[d]["located"]["q25"] if rows[d]["all_runs_located"] else np.nan for d in xs]
            upper = [rows[d]["located"]["q75"] if rows[d]["all_runs_located"] else np.nan for d in xs]
            ax.plot(xs, vals, marker=marker, ms=3.2, lw=1.15, color=colors[instrument],
                    mfc="white" if instrument == "checkpoint" else colors[instrument])
            ax.fill_between(xs, lower, upper, color=colors[instrument], alpha=.12)
        if mean:
            rows = sorted((r for r in data["checkpoint"] if r["alpha"] == alpha and r["L"] == depth), key=lambda r:r["d"])
            ax.plot([r["d"] for r in rows], [r["mean_curve_crossing"] if not r["n_censored"] and r["mean_curve_crossing"] else np.nan for r in rows],
                    "^:", ms=3.5, lw=1.2, color=colors["mean"])
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlim(52, 41000)
        ax.set_xticks([64, 256, 1024, 4096, 16384])
        ax.set_xticklabels(["64", "256", "1,024", "4,096", "16,384"], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, which="major", alpha=.16)
        fitted = fits["refined_median", "primary", alpha, depth]["fit"]
        suffix = rf", $p={fitted['p']:.2f}$" if fitted else ""
        ax.set_title(rf"$\alpha={alpha:g}$" + suffix, fontsize=10)
        if not any(line.get_ydata().size and np.any(np.isfinite(line.get_ydata())) for line in ax.lines):
            ax.set_ylim(1, 1000)
            ax.text(.5, .5, "No setting has a located\ncrossing in every run", ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#666")

    legend = [Line2D([0], [0], color=colors["refined"], marker="o", ms=4, label="Refined bracket endpoints: median"),
              Line2D([0], [0], color=colors["checkpoint"], marker="s", mfc="white", ms=4, label="Checkpoint search: median"),
              Line2D([0], [0], color=colors["mean"], ls=":", marker="^", ms=4, label="Crossing of average costs (16 runs)")]
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        # This compares growth exponents across parameters, instead of repeating L=3.
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3))
        depth_colors = ["#176c91", "#2c885f", "#c16a18", "#993b65"]
        for depth, color in zip([2, 3, 4, 6], depth_colors):
            vals = [fits["refined_median", "primary", a, depth]["fit"]["p"] for a in alphas]
            axes[0].plot(alphas, vals, "o-", lw=1.5, ms=4, color=color, label=f"Source L = {depth}")
        axes[0].set_xlabel(r"Source parameter $\alpha$")
        axes[0].set_title("Fitted exponent across source parameters", fontsize=12)
        axes[0].legend(frameon=False, fontsize=9, loc="lower left")
        chosen = [.5, .75, 1., 1.1, 1.25, 1.5]
        for alpha, color in zip(chosen, plt.get_cmap("tab10").colors):
            local = fits["refined_median", "primary", alpha, 3]["fit"]["local_adjacent_slopes"]
            axes[1].plot([r["to_d"] for r in local], [r["slope"] for r in local], "o-",
                         color=color, lw=1.3, ms=4, label=rf"$\alpha={alpha:g}$")
        axes[1].set_xscale("log", base=2)
        axes[1].set_xticks([8192, 16384, 32768], ["4,096 to\n8,192", "8,192 to\n16,384", "16,384 to\n32,768"])
        axes[1].set_xlim(7000, 39000)
        axes[1].set_xlabel("Alphabet doubling")
        axes[1].set_title("Exponent from each doubling, source L = 3", fontsize=12)
        axes[1].legend(frameon=False, fontsize=9, ncol=3, loc="lower left")
        for ax in axes:
            ax.set_ylim(.25, 2.1)
            ax.set_yticks([.5, 1, 1.5, 2])
            ax.set_ylabel("Measured exponent p")
            ax.grid(True, alpha=.15)
            for p in [1, 1.5, 2]:
                ax.axhline(p, color="#777", lw=.7, ls=":", zorder=0)
        fig.suptitle("What the straight log-log curves tell us", x=.06, ha="left", y=.97, fontsize=19)
        fig.text(.06, .905, r"A power law $n\approx A d^p$ is a straight line in log-log coordinates. Its exponent is $p$.", fontsize=12)
        fig.text(.06, .855, "Left: compare the measured exponents. Right: check whether they change as the alphabet grows.", fontsize=11)
        fig.text(.075, .17, "p = 2: doubling d multiplies n by 4.     p = 1.5: by 2.83.     p = 1: by 2.", fontsize=11)
        footer(fig, "Fits use d = 4,096, 8,192, 16,384, 32,768 and refined median candidate crossings. They describe this measured range.\nPage 2 explains the measurement flags. Pages 3-12 contain the underlying curves, once per source depth.\nLater reversals remain a known search limitation. The rich-text report compares these results directly with the theory.")
        page_number(fig, 1)
        fig.subplots_adjust(left=.075, right=.97, bottom=.30, top=.77, wspace=.28)
        pdf.savefig(fig)
        fig.savefig(out.with_suffix(".png"), dpi=160)
        plt.close(fig)

        # Overview of the full source-parameter range at two alphabet sizes.
        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        for col, d in enumerate([1024, 32768]):
            med = np.full((len(depths), len(alphas)), np.nan)
            early = np.zeros_like(med)
            lookup = {(r["alpha"], r["L"]): r for r in data["refined"] if r["d"] == d}
            for i, L in enumerate(depths):
                for j, a in enumerate(alphas):
                    r = lookup.get((a, L))
                    if r:
                        if r["all_runs_located"]:
                            med[i, j] = math.log10(r["located"]["median"])
                        early[i, j] = r["n_positive_first_check"] / r["n_seeds"]
            ax = axes[0, col]
            cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#e8e8e8")
            im = ax.imshow(med, cmap=cmap, vmin=1, vmax=8, aspect="auto")
            cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.025, ticks=[1, 3, 5, 7])
            cb.ax.set_yticklabels(["10", r"$10^3$", r"$10^5$", r"$10^7$"])
            cb.ax.tick_params(labelsize=8)
            ax.set_title(f"d = {d:,}: median crossing", fontsize=12)
            ax = axes[1, col]
            im = ax.imshow(early, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=.035, pad=.025, ticks=[0, .5, 1]).ax.set_yticklabels(["0%", "50%", "100%"])
            ax.set_title("Runs positive at the first check (8 tokens)", fontsize=11)
            for ax in axes[:, col]:
                ax.set_xticks(range(len(alphas)), [f"{a:g}" for a in alphas], fontsize=8)
                ax.set_yticks(range(len(depths)), depths, fontsize=8)
                ax.set_ylabel("Source depth L")
                ax.set_xlabel(r"$\alpha$")
        fig.suptitle("Which source settings yield a measurable crossing?", x=.06, ha="left", y=.975, fontsize=17)
        fig.text(.06, .925, "Upper panels: located crossings. Lower panels: early-positive flags, kept separate.", fontsize=11)
        footer(fig, "Grey cells in the upper panels contain at least one unlocated early-positive run; they are not assigned a median crossing.\nA positive first check does not exclude a later reversal. Source L and alpha are parameters of the generated data.")
        fig.subplots_adjust(left=.075, right=.955, bottom=.13, top=.85, wspace=.34, hspace=.45)
        page_number(fig, 2)
        pdf.savefig(fig); plt.close(fig)

        # Every core source setting, with both observation methods kept separate.
        for number, depth in enumerate(depths, 3):
            fig, axes = plt.subplots(5, 2, figsize=(8.3, 11.7))
            for ax, alpha in zip(axes.flat, alphas):
                panel(ax, alpha, depth, mean=True)
                ax.set_ylim(1, 2e8)
                ax.set_yticks([1, 100, 10000, 1000000, 100000000])
                ax.yaxis.set_minor_formatter(NullFormatter())
            fig.suptitle(f"Crossing length versus alphabet size | source L = {depth}", x=.075, ha="left", y=.98, fontsize=14)
            fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .957), ncol=2, frameon=False, fontsize=8)
            fig.supxlabel("Alphabet size d", y=.077); fig.supylabel("Reported crossing length (tokens)", x=.012)
            footer(fig, "Same axes on all data pages. Panel p: fitted exponent from the four largest alphabet sizes.\nBands: middle half of runs; blue/orange require a located crossing in every run. Green: average costs.\nRefined runs: 256 / 128 / 64 / 32 as d grows (four settings: 16). Some brackets precede later reversals.")
            page_number(fig, number)
            fig.subplots_adjust(left=.15, right=.97, top=.915, bottom=.14, hspace=.63, wspace=.3)
            pdf.savefig(fig); plt.close(fig)
    print(f"Wrote {out} (12 pages) and first-page PNG")


def rich_text_report():
    """Export the reviewed prose and tables as an editable, ordinary RTF."""
    source = REPO / "notes/reviews/mechanisms_section4_observations_2026-09-06.md"
    def escape(text):
        parts = []
        for ch in text:
            if ch in "\\{}":
                parts.append("\\" + ch)
            elif ord(ch) > 127:
                code = ord(ch)
                parts.append("\\u" + str(code if code < 32768 else code - 65536) + "?")
            else:
                parts.append(ch)
        return "".join(parts)
    out = [r"{\rtf1\ansi\deff0{\fonttbl{\f0 Helvetica;}}",
           r"\paperw11906\paperh16838\margl1080\margr1080\margt1080\margb1080",
           r"\widowctrl\f0\fs22"]
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("| ---"):
            continue
        if line.startswith("|"):
            cells = [s.strip() for s in line.strip("|").split("|")]
            width = 9746 // len(cells)
            out.append(r"\trowd\trgaph100\trleft0")
            for i in range(len(cells)):
                out.append(r"\clbrdrb\brdrs\brdrw5\cellx" + str(width * (i+1)))
            for cell in cells:
                out.append(r"\pard\intbl\fs20 " + escape(cell) + r"\cell")
            out.append(r"\row\pard\fs22")
        elif line.startswith("# "):
            out.append(r"\pard\keepn\sb0\sa220\fs34\b " + escape(line[2:]) + r"\b0\fs22\par")
        elif line.startswith("## "):
            out.append(r"\pard\keepn\sb240\sa100\fs26\b " + escape(line[3:]) + r"\b0\fs22\par")
        else:
            out.append(r"\pard\sb0\sa130\sl275\slmult1 " + escape(line) + r"\par")
    out.append("}")
    target = source.with_suffix(".rtf")
    target.write_text("\n".join(out), encoding="ascii")
    print(f"Wrote editable report: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--pdf-out", type=Path)
    args = parser.parse_args()
    if not (args.collect or args.check or args.plot or args.report):
        parser.error("Choose --collect, --check, --plot or --report")
    if args.collect or args.check:
        result = collect()
        if args.check:
            # Normalize integer JSON object keys used in the repetition counts.
            assert json.loads(json.dumps(result)) == json.loads(args.out.read_text()), "Archive differs from original inputs"
            print("Every source hash, count, summary and mean-curve crossing verified")
        if args.collect:
            save(result, args.out)
            print(json.dumps(result["summary"], indent=2))
    if args.plot:
        plot(json.loads(args.out.read_text()), args.pdf_out or args.out.with_suffix(".pdf"))
    if args.report:
        rich_text_report()


if __name__ == "__main__":
    main()
