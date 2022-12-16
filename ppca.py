import numpy as np


def relu(x):
    return np.where(x > 0, x, 0)


class PPCA:
    def __init__(self, configs, X=None):
        self.configs = configs
        self.q = configs.model_configs.latent_dim
        self.d = configs.model_configs.data_dim
        self.N = configs.model_configs.data_size
        self.mle_sigma_sq = None
        self.mle_W = None
        self.X = X

    def set_latent_dim(self, latent_dim):
        self.q = latent_dim

    def calculate_SCM(self, X):
        n, p = X.shape
        x_mean = np.mean(X, axis=0, keepdims=True)
        SCM = (X - x_mean).T @ (X - x_mean) / (n - 1)
        return SCM

    def ppca(self, SCM, latent_dim):
        d, q = SCM.shape[0], latent_dim
        eig_vals, eig_vecs = np.linalg.eig(SCM)
        eig_vals = eig_vals.real
        eig_vecs = eig_vecs.real
        eig_sorted_index = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[eig_sorted_index]
        eig_vecs = eig_vecs[:, eig_sorted_index]
        mle_sigma_sq = np.sum(eig_vals[q:]) / (d - q)
        Uq = eig_vecs[:, : q]
        mle_W = Uq @ ((np.diag(eig_vals[: q]) - np.eye(q) * mle_sigma_sq) ** 0.5)
        return mle_sigma_sq, mle_W

    def ppca_fix_sigma_dim(self, SCM, latent_dim, s_dim=50):
        d, q = SCM.shape[0], latent_dim
        eig_vals, eig_vecs = np.linalg.eig(SCM)
        eig_vals = eig_vals.real
        eig_vecs = eig_vecs.real
        eig_sorted_index = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[eig_sorted_index]
        eig_vecs = eig_vecs[:, eig_sorted_index]
        fix_sigma_sq = np.sum(eig_vals[s_dim:]) / (d - s_dim)
        Uq = eig_vecs[:, : q]
        mle_W = Uq @ (relu(np.diag(eig_vals[: q]) - np.eye(q) * fix_sigma_sq) ** 0.5)
        return fix_sigma_sq, mle_W

    def log_likelihood(self, W, sigma_sq, SCM, N):
        d = SCM.shape[0]
        C = W @ W.T + sigma_sq * np.eye(d)
        C_inv = np.linalg.inv(C)
        L = - N / 2 * (d * np.log(2 * np.pi) + np.linalg.slogdet(C)[1] + np.trace(C_inv @ SCM))
        return L

    def fit(self, X):
        self.X = X
        self.N = X.shape[0]
        self.SCM = self.calculate_SCM(X)
        self.mle_sigma_sq, self.mle_W = self.ppca(self.SCM, self.q)
        self.log_lik = self.log_likelihood(self.mle_W, self.mle_sigma_sq, self.SCM, self.N)
        self.avg_log_lik = self.log_lik / self.N
        return self.avg_log_lik

    def fit_fix_sigma(self, X, s_dim=50):
        self.X = X
        self.N = X.shape[0]
        self.SCM = self.calculate_SCM(X)
        self.mle_sigma_sq, self.mle_W = self.ppca_fix_sigma_dim(self.SCM, self.q, s_dim)
        self.log_lik = self.log_likelihood(self.mle_W, self.mle_sigma_sq, self.SCM, self.N)
        self.avg_log_lik = self.log_lik / self.N
        return self.avg_log_lik
