"""Step 5 library: one-row quantities from sampled rows and the second-order self-loop theory of the chain inputs."""
import numpy as np
import lsa_cost as C


class RowTheory:
    def __init__(self, d, alpha, Ls, NR=100000, seed=2024):
        self.d, self.alpha, self.Ls = d, alpha, Ls
        q = C.zipf(d, alpha); self.q = q
        rr = np.random.default_rng(seed)
        P = np.empty((NR, d)); Hr = np.empty(NR)
        for s in range(0, NR, 5000):
            W = np.prod(rr.exponential(size=(Ls, min(5000, NR - s), d)), axis=0); Pb = q[None, :] * W; Pb /= Pb.sum(axis=1, keepdims=True)
            P[s:s + Pb.shape[0]] = Pb; Hr[s:s + Pb.shape[0]] = -np.sum(Pb * np.log2(Pb), axis=1)
        self.NR = NR; self.P = P; self.Hr = Hr
        self.mu = P.mean(axis=0); self.x = P - self.mu
        inv = 1.0 / (1.0 - self.x); y = self.x * inv
        self.S = (inv.T @ self.x) / NR                       # S[i,a] = E[x_a/(1-x_i)]
        self.M2 = ((inv ** 2).T @ (self.x ** 2)) / NR - self.S ** 2   # Var(x_a/(1-x_i))
        self.s_diag = np.diag(self.S).copy()
        self.cinf = (y * Hr[:, None]).mean(axis=0) - y.mean(axis=0) * Hr.mean()   # Cov(x_a/(1-x_a), H)
        self.Hbar_rows = Hr.mean()

    def trunc_sums(self, M):
        x = self.x; one = np.abs(1.0 - x) < 1e-9
        xs = np.where(one, 0.5, x)
        B1 = (1.0 - xs ** (M + 1)) / (1.0 - xs)
        A = (M + 2) * B1 - (1.0 - (M + 2) * xs ** (M + 1) + (M + 1) * xs ** (M + 2)) / (1.0 - xs) ** 2
        B = B1 + xs ** (M + 1)
        B = np.where(one, M + 2.0, B); A = np.where(one, (M + 2) * (M + 1) / 2.0, A)
        return A, B

    def inputs(self, n, finite=False):
        """Returns dict with EK, VK (total), EN, VN (total), cov, VaK (across-T part), VwK (within part)."""
        q, mu, S, M2, s_diag, cinf, x, NR = self.q, self.mu, self.S, self.M2, self.s_diag, self.cinf, self.x, self.NR
        if not finite:
            corr = mu @ S; qcorr = q @ S
            EK = q + (n - 2) * (mu + corr) + qcorr
            EN = q + (n - 1) * (mu + corr) + qcorr
            v = mu ** 2 @ M2
            VaK = (n - 2) ** 2 * v; VaN = (n - 1) ** 2 * v
        else:
            A, B = self.trunc_sums(n - 4)
            SA = (A.T @ x) / NR; SB = (B.T @ x) / NR
            EK = q + (n - 2) * mu + mu @ SA + q @ SB
            VaK = mu ** 2 @ (((A ** 2).T @ (x ** 2)) / NR - SA ** 2)
            A, B = self.trunc_sums(n - 3)
            SA = (A.T @ x) / NR; SB = (B.T @ x) / NR
            EN = q + (n - 1) * mu + mu @ SA + q @ SB
            VaN = mu ** 2 @ (((A ** 2).T @ (x ** 2)) / NR - SA ** 2)
            v = VaK / (n - 2) ** 2
        VwK = EK - (q ** 2 + (n - 2) * (mu ** 2 + v)) + 2 * (n - 1) * (1 + mu) * s_diag
        VwN = EN - (q ** 2 + (n - 1) * (mu ** 2 + v)) + 2 * n * (1 + mu) * s_diag
        cov = (n - 2) * np.sum(mu * cinf) + np.sum(q * cinf)
        return dict(EK=EK, VK=np.maximum(VwK, 0) + VaK, EN=EN, VN=np.maximum(VwN, 0) + VaN, cov=cov, VaK=VaK, VwK=VwK, VaN=VaN, VwN=VwN)
