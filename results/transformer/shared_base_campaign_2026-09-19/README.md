# Shared-base capacity and training release

Both four-layer continuations and eight-layer models completed 400k updates
(25.6 million training sequences), two runs and ten milestone evaluations.
The diagnostic cohort has256 sequences per L=0,1,4,8, alpha=1, at
1,4,16,64,256,1024 tokens; it is not an independent confirmation sample.
This historical release supplies Bayes only at1,4,16,64. Later references
are separate releases. Runs are never averaged or selected.

`comparison.json`, `endpoints.tsv` and `validation.json` retain all scores,
source hashes, paired differences and checks. Error bars are one source SE,
not training or numerical uncertainty. Saved scores/provenance were audited;
checkpoint inference was not replayed. No smoothing or normalization repair.

Reproduce from the repository root with pinned raw inputs available:
```
python code/scripts/shared_base_campaign_results.py --input output/kuma_results_2026-09-20/shared_base_campaign_2026-09-18
```
Use `--no-plot --out /tmp/shared-base-campaign-audit` for a separate audit.
Tests: `python -m pytest -q code/tests/test_shared_base_campaign_results.py`.
`load_campaign(source)` supplies the figure interface.

The full original README (including validation scope, results, exact commands
and input paths) is preserved byte-for-byte as
`before/results/transformer/shared_base_campaign_2026-09-19/README.md` in
`notes/archives/overleaf_merge_2026-09-25_1949.tar.gz`.
