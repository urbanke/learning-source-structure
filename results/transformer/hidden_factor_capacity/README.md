# Hidden-factor capacity benchmark

`benchmark-4233974.json` preserves the JSON report pasted by Ruediger from
Kuma job 4233974, run at repository commit 42ae47d. The job completed in
5 minutes 11 seconds with exit code 0:0. It used one NVIDIA H100, float64,
and two independent length-1025 sequence timings per feasible candidate.
The report's three implementation hashes match the repository; encoding
counts and all runtime projections were independently checked on import.

| Dimensions | Tokens | Exact encodings | Projected reference time, 4096 sequences |
|---|---:|---:|---:|
| 2x5 | 10 | 15,120 | 0.019 hours |
| 2x6 | 12 | 332,640 | 0.055 hours |
| 3x4 | 12 | 3,326,400 | 0.333 hours |
| 2x7 | 14 | 8,648,640 | 1.021 hours |
| 2x8 | 16 | 259,459,200 | 35.883 hours |

The 3x5 and 2x9 cases were rejected by the declared 8 GiB encoding-cache
limit before timing. This is a resource-policy rejection, not a mathematical
impossibility result.

The automatic recommendation is 2x7 because the benchmark used an initial
24-hour planning cutoff. The larger 2x8 case completed both repetitions;
its 4.15 GB cache was built in 169 seconds and peak GPU allocation was
0.512 GB. Given the user's preference for the largest practical source,
2x8 was recommended and the user approved proceeding after the runtime
estimate was explained. Next is the training-throughput check. The full
reference cost is an estimate, not completed work. No hidden-factor training
or scientific evaluation result is supplied by this benchmark.
