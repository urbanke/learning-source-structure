# Hidden-factor pilot: 2 × 8, 256 test sequences

These results are the first 256 sequences of the frozen 4096-sequence test
cohort, each of length 1025. Both transformers were trained on 6.4 million
fresh sequences. Their parameters and joint token encodings are drawn anew
for every sequence. Training is complete; the full test cohort is not.

The independently audited pilot shows early agreement but a substantial
long-context gap from informed Bayes. At 1024 observations:

| Predictor | Expected next-token regret (bits) |
|---|---:|
| Bayes, encoding hidden | 0.03705 |
| Bayes, encoding given | 0.03651 |
| Transformer, seed 0 | 0.14651 |
| Transformer, seed 1 | 0.13506 |

The transformers' cumulative excess over Bayes is 107.31 ± 2.29 and
104.98 ± 2.26 bits, respectively. The cumulative cost to Bayes of the hidden
encoding is 27.54 ± 0.10 bits. These are different quantities. Uncertainty
is one standard error across independent sequences, pairing predictor
differences within each sequence. This result describes the tested models
and training budget; it does not establish that no structure was learned.

## Provenance and validation

- `summary.json` is a byte-identical copy of the cluster pilot summary.
- `validation.json` binds that summary and the independent validator, and
  records the audit of 768 saved sequence-level records and 1547 file hashes.
- `validate_pilot.py` reconstructs the source, known-encoding and flat-Dirichlet
  references independently, checks predictive/evidence identities, and
  recomputes all means and standard errors. It does not repeat full hidden
  encoding enumeration or transformer inference. Actual checkpoint binaries
  were not transferred; distinct recorded hashes and final steps were checked.
- `figures/` copies the verified cluster figures and their receipt. The
  uncertainty-cost figure averages the full pilot; the conditional figure
  retains all four predefined entropy groups (68, 57, 62, 69 sequences).

Raw inputs remain outside Git in
`output/hidden_factor_cluster_pilot_2026-09-16/`. User-reported Slurm job
4234386 completed in 2 h 12 min 16 s, with exit code 0:0. No experiment was
rerun to make these manuscript figures.

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python results/transformer/hidden_factor_pilot_2026-09-16/validate_pilot.py --self-test
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/scripts/hidden_factor_figures.py --summary results/transformer/hidden_factor_pilot_2026-09-16/summary.json --out results/transformer/hidden_factor_pilot_2026-09-16/figures --check-only
```

The first command requires the transferred raw inputs. The second verifies
the committed summary, plotting code and all saved figure hashes.
