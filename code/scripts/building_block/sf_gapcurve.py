import sys, json, numpy as np
from sf_predict import SFPredict
d, a, L, bbf = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ns = [int(x) for x in sys.argv[5].split(",")]
sf = SFPredict(d, a, L, bbf, ns, verbose=False)
for n in ns:
    C, cov, EK, VA = sf.copier(n); F = sf.unigram(n)
    print(f"n={n:6d}: copier {C+cov:9.1f} unigram {F:9.1f} gap {F-C-cov:+8.2f}", flush=True)
