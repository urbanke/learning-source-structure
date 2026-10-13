"""Information gain per token I_T = E[H(pi_T)] - E[H(row)]: MC vs explicit formulas."""
import numpy as np, sys, time
from scipy.special import logsumexp
import lsa_cost as C, source_profiles as S

EULER = 0.57721566490153286


def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def stationary(T):
    # solve pi T = pi
    d = T.shape[0]
    A = T.T - np.eye(d)
    A[-1, :] = 1.0
    b = np.zeros(d); b[-1] = 1.0
    return np.linalg.solve(A, b)


def mc_info(d, alpha, Ls, ntab, rng):
    q = C.zipf(d, alpha)
    Hrow, Hpi, I = [], [], []
    for _ in range(ntab):
        W = np.prod(rng.exponential(size=(Ls, d, d)), axis=0)
        T = q[None, :] * W
        T /= T.sum(axis=1, keepdims=True)
        pi = stationary(T)
        hrows = np.array([entropy(T[a]) for a in range(d)])
        Hrow.append(hrows.mean())
        Hpi.append(entropy(pi))
        I.append(entropy(pi) - np.dot(pi, hrows))
    Hrow, Hpi, I = map(np.array, (Hrow, Hpi, I))
    n = ntab
    return (Hrow.mean(), Hrow.std(ddof=1) / np.sqrt(n * d), Hpi.mean(), Hpi.std(ddof=1) / np.sqrt(n),
            I.mean(), I.std(ddof=1) / np.sqrt(n))


def second_order(q, Ls):
    v = 2.0 ** Ls - 1.0
    H = entropy(q)
    s2 = np.sum(q ** 2)
    s2l = np.sum(q ** 2 * np.log(q))
    EHP = (H - Ls * (1 - EULER) + v * (s2l + s2 * H)
           + s2 * (Ls * (2.0 ** Ls * (1.5 - EULER) - (1 - EULER)) - v / 2 - v * Ls * (1 - EULER)))
    EHpi = H + v * (s2l + s2 * H) - 0.5 * v * s2
    I = Ls * (1 - EULER) - Ls * 2.0 ** (Ls - 1) * s2
    return EHP, EHpi, I


def exact_EHP(q, Ls):
    """E H(P) by the Gamma-trick integrals (exact up to quadrature)."""
    y, lfh = S.lnW_density(Ls, ymin=-40.0, h=1.0/32)
    W = np.exp(y)
    lnW = y
    s = np.exp(np.arange(-12.0, 9.0, 0.04))            # s grid (log-spaced)
    ds = np.gradient(s)
    # ln phi_0(s q_c) for all c: matrix (d, ns)
    lq = np.log(q)
    lphi0 = np.array([logsumexp(lfh[:, None] - (qc * s)[None, :] * W[:, None], axis=0) for qc in q])
    lnG = lphi0.sum(axis=0)
    G = np.exp(lnG)
    # E ln Z = int (e^{-s} - G(s))/s ds
    ElnZ = np.sum((np.exp(-s) - G) / s * ds)
    # E[q_b W_b ln(q_b W_b) / Z] = int q_b [ln q_b phi_1(s q_b) + psi(s q_b)] G_{-b}(s) ds
    tot = 0.0
    for b, qb in enumerate(q):
        lphi1 = logsumexp(lfh[:, None] + lnW[:, None] - (qb * s)[None, :] * W[:, None], axis=0)
        # psi(t) = E[W ln W e^{-tW}]  (W ln W can be negative for W<1: split by sign)
        pos = lnW > 0
        lpsi_pos = logsumexp(lfh[pos, None] + lnW[pos, None] + np.log(lnW[pos])[:, None] - (qb * s)[None, :] * W[pos, None], axis=0)
        lpsi_neg = logsumexp(lfh[~pos, None] + lnW[~pos, None] + np.log(-lnW[~pos])[:, None] - (qb * s)[None, :] * W[~pos, None], axis=0)
        psi = np.exp(lpsi_pos) - np.exp(lpsi_neg)
        Gmb = np.exp(lnG - lphi0[b])
        tot += np.sum(qb * (lq[b] * np.exp(lphi1) + psi) * Gmb * ds)
    return ElnZ - tot


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    print("d alpha Ls | MC: E H(row) (se)  E H(pi) (se)  I (se) | 2nd order: EHP EHpi I | exact EHP | H(q)-Ls(1-g)")
    cases = [(256, 0.75, 2, 100), (256, 0.5, 2, 100), (256, 1.0, 2, 100), (256, 0.75, 1, 100), (256, 0.75, 3, 100), (128, 0.75, 2, 200), (512, 0.75, 2, 40), (1024, 0.5, 3, 15), (1024, 1.0, 3, 15)]
    if len(sys.argv) > 1: cases = [cases[int(a)] for a in sys.argv[1:]]
    for d, alpha, Ls, ntab in cases:
        q = C.zipf(d, alpha)
        t0 = time.time()
        res = mc_info(d, alpha, Ls, ntab, rng)
        so = second_order(q, Ls)
        ex = exact_EHP(q, Ls)
        print(f"{d} {alpha} {Ls} | {res[0]:.4f} ({res[1]:.4f}) {res[2]:.4f} ({res[3]:.4f}) {res[4]:.4f} ({res[5]:.4f}) | {so[0]:.4f} {so[1]:.4f} {so[2]:.4f} | {ex:.4f} | {entropy(q) - Ls*(1-EULER):.4f}   [{time.time()-t0:.0f}s]", flush=True)
