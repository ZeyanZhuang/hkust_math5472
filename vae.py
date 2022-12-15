import torch
import torch.nn as nn
import numpy as np

LOG2PI = np.log(2 * np.pi)


class LinearVAE(nn.Module):
    def __init__(self, configs) -> None:
        super(LinearVAE, self).__init__()

        # Configs
        self.alg_configs = configs.alg_configs
        self.model_configs = configs.model_configs
        self.x_dim = self.model_configs.data_dim
        self.z_dim = self.model_configs.latent_dim
        self.device = self.model_configs.device

        # Encoder
        self.VT = nn.Parameter(torch.randn(self.x_dim, self.z_dim) / np.sqrt(self.x_dim))
        self.log_D = nn.Parameter(torch.randn(self.z_dim))
        self.mu = nn.Parameter(torch.randn(1, self.x_dim))

        # Decoder
        self.WT = nn.Parameter(torch.randn(self.z_dim, self.x_dim) / np.sqrt(self.z_dim))
        if self.model_configs.init_sigma is not None:
            init_log_sigma_tensor = torch.tensor(np.log(self.model_configs.init_sigma ** 2), dtype=torch.float32)
        else:
            init_log_sigma_tensor = torch.randn(1)
        self.log_sigma_sq = nn.Parameter(init_log_sigma_tensor, requires_grad=self.model_configs.train_sigma)
        self.to(self.device)

    def encoder_step(self, X):
        batch_size = X.shape[0]
        self.z_mean = z_mean = torch.matmul(X - self.mu, self.VT)
        rnd_noise = torch.randn(batch_size, self.z_dim).to(self.device)
        cov_squ = torch.diag(torch.exp(self.log_D * 0.5))
        z = z_mean + torch.matmul(rnd_noise, cov_squ)
        return z_mean, z

    def decoder_step(self, Z):
        self.x_mean = x_mean = torch.matmul(Z, self.WT) + self.mu
        return x_mean

    def explicit_recons(self, X):
        VTWT = torch.matmul(self.VT, self.WT)
        D = torch.diag(torch.exp(self.log_D))
        WDWT = self.WT.T @ D @ self.WT
        X_ = X - self.mu
        XTVTWT = X_ @ VTWT

        item_1 = -torch.trace(WDWT)
        item_2 = -torch.sum(XTVTWT ** 2, dim=1)
        item_3 = 2 * torch.sum(X_ * XTVTWT, dim=1)
        item_4 = - torch.sum(X_ ** 2, dim=1)

        sigma_sq = torch.exp(self.log_sigma_sq)
        rec_loss = 0.5 * (
                (item_1 + item_2 + item_3 + item_4) / sigma_sq
                - self.x_dim * (LOG2PI + self.log_sigma_sq)
        )
        return torch.mean(rec_loss)

    def implicit_recons(self, X):
        z_mean, z = self.encoder_step(X)
        x_mean = self.decoder_step(z)
        sigma_sq = torch.exp(self.log_sigma_sq)
        rec_loss = 0.5 * (
                - torch.sum((X - x_mean) ** 2, dim=1) / sigma_sq
                - self.x_dim * (LOG2PI + self.log_sigma_sq)
        )
        return torch.mean(rec_loss)

    def KL_to_gaussian_loss(self, X):
        Tr_D = torch.sum(torch.exp(self.log_D))
        log_det_D = torch.sum(self.log_D)
        XTVT = torch.matmul(X - self.mu, self.VT)
        item_var = torch.sum(XTVT ** 2, dim=1)
        KL = 0.5 * (Tr_D - log_det_D + item_var - self.z_dim)
        return torch.mean(KL)

    def ELBO(self, X):
        recons_loss = 0
        if self.model_configs.explicit_mode:
            recons_loss = self.explicit_recons(X)
        else:
            recons_loss = self.implicit_recons(X)
        kl_loss = self.KL_to_gaussian_loss(X)
        self.elbo = recons_loss - kl_loss
        return self.elbo

    def forward(self, inputs):
        cost = -self.ELBO(X=inputs)
        return cost

    def posterior_collapse_rate(self, X, eps, delta=0.01):
        batch_size = X.shape[0]
        with torch.no_grad():
            z_mean = torch.matmul(X - self.mu, self.VT)
            z_log_sigma_sq = self.log_D.unsqueeze(0)
            z_sigma_sq = torch.exp(z_log_sigma_sq)
            KL_element_wise = 0.5 * (
                    z_sigma_sq
                    - z_log_sigma_sq
                    + z_mean ** 2
                    - 1
            )
            z_posterior_collapse = torch.where(KL_element_wise < eps, 1, 0)
            prob_x_pc = torch.sum(z_posterior_collapse, dim=0) / batch_size
            z_collapse = torch.where(prob_x_pc >= 1 - delta, 1, 0)
            collapsed_rate = torch.sum(z_collapse) / self.z_dim
        return collapsed_rate.item()


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.x_dim = input_dim
        self.z_dim = output_dim
        self.fc_1 = nn.Linear(input_dim, hidden_dim[0])
        self.act_1 = nn.ReLU(inplace=True)
        self.fc_2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.act_2 = nn.ReLU(inplace=True)
        self.out_mu = nn.Linear(hidden_dim[1], self.z_dim)
        self.out_log_sigma_sq = nn.Linear(hidden_dim[1], self.z_dim)

    def forward(self, x):
        x = self.act_1(self.fc_1(x))
        shared_feature = self.act_2(self.fc_2(x))
        self.mu = self.out_mu(shared_feature)
        self.log_sigma_sq = self.out_log_sigma_sq(shared_feature)
        return self.mu, self.log_sigma_sq


class DeepLinearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepLinearEncoder, self).__init__()
        self.x_dim = input_dim
        self.z_dim = output_dim
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.out_mu = nn.Linear(hidden_dim, self.z_dim)
        self.out_log_sigma_sq = nn.Linear(hidden_dim, self.z_dim)

    def forward(self, x):
        shared_feature = self.fc_1(x)
        self.mu = self.out_mu(shared_feature)
        self.log_sigma_sq = self.out_log_sigma_sq(shared_feature)
        return self.mu, self.log_sigma_sq


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, init_sigma, train_sigma):
        super(Decoder, self).__init__()
        self.z_dim = input_dim
        self.x_dim = output_dim
        self.fc_1 = nn.Linear(input_dim, hidden_dim[0])
        self.act_1 = nn.ReLU(inplace=True)
        self.fc_2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.act_2 = nn.ReLU(inplace=True)
        self.out_x = nn.Linear(hidden_dim[1], output_dim)

        if init_sigma is not None:
            init_log_sigma_tensor = torch.tensor(np.log(init_sigma ** 2), dtype=torch.float32)
        else:
            init_log_sigma_tensor = torch.randn(1)
        self.log_sigma_sq = nn.Parameter(init_log_sigma_tensor, requires_grad=train_sigma)

    def forward(self, z):
        z = self.act_1(self.fc_1(z))
        z = self.act_2(self.fc_2(z))
        x_mean = self.out_x(z)
        return x_mean


class LinearDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, init_sigma, train_sigma):
        super(LinearDecoder, self).__init__()
        self.z_dim = input_dim
        self.x_dim = output_dim
        self.out_x = nn.Linear(input_dim, output_dim)
        if init_sigma is not None:
            init_log_sigma_tensor = torch.tensor(np.log(init_sigma ** 2), dtype=torch.float32)
        else:
            init_log_sigma_tensor = torch.randn(1)
        self.log_sigma_sq = nn.Parameter(init_log_sigma_tensor, requires_grad=train_sigma)

    def forward(self, z):
        x_mean = self.out_x(z)
        return x_mean


class DeepVAE(nn.Module):
    def __init__(self, configs) -> None:
        super(DeepVAE, self).__init__()
        # Configs
        self.alg_configs = configs.alg_configs
        self.model_configs = configs.model_configs
        self.x_dim = self.model_configs.data_dim
        self.z_dim = self.model_configs.latent_dim
        self.device = self.model_configs.device
        # Encoder
        if self.model_configs.encoder_type == 'mlp':
            self.encoder = Encoder(
                self.x_dim, [1024, 512], self.z_dim
            )
        elif self.model_configs.encoder_type == 'deep_linear':
            self.encoder = DeepLinearEncoder(
                self.x_dim, 512, self.z_dim
            )
        # Decoder
        if self.model_configs.decoder_type == 'linear':
            self.decoder = LinearDecoder(
                self.z_dim, self.x_dim, self.model_configs.init_sigma, self.model_configs.train_sigma
            )
        elif self.model_configs.decoder_type == 'mlp':
            self.decoder = Decoder(
                self.z_dim, [512, 1024], self.x_dim, self.model_configs.init_sigma, self.model_configs.train_sigma
            )
        elif self.model_configs.decoder_type == 'conv':
            pass

        self.to(self.device)

    def encoder_step(self, X):
        batch_size = X.shape[0]
        z_mean, z_log_sigma_sq = self.encoder(X)
        rnd_noise = torch.randn(batch_size, self.z_dim).to(self.device)
        z_sigma = torch.exp(z_log_sigma_sq * 0.5)
        z = z_mean + rnd_noise * z_sigma
        return z_mean, z

    def decoder_step(self, Z):
        x_mean = self.decoder(Z)
        return x_mean

    def implicit_recons(self, X):
        z_mean, z = self.encoder_step(X)
        x_mean = self.decoder_step(z)
        sigma_sq = torch.exp(self.decoder.log_sigma_sq)
        rec_loss = 0.5 * (
                - torch.sum((X - x_mean) ** 2, dim=1) / sigma_sq
                - self.x_dim * (LOG2PI + self.decoder.log_sigma_sq)
        )
        return torch.mean(rec_loss)

    def KL_to_gaussian_loss(self):
        z_mean, z_log_sigma_sq = self.encoder.mu, self.encoder.log_sigma_sq
        Tr_S = torch.sum(torch.exp(z_log_sigma_sq), dim=1)
        log_det_S = torch.sum(z_log_sigma_sq, dim=1)
        item_var = torch.sum(z_mean ** 2, dim=1)
        KL = 0.5 * (Tr_S - log_det_S + item_var - self.z_dim)
        return torch.mean(KL)

    def ELBO(self, X):
        # sample loss
        recons_loss = self.implicit_recons(X)
        # KL divergence
        kl_loss = self.KL_to_gaussian_loss()
        self.elbo = recons_loss - kl_loss
        return self.elbo

    def forward(self, inputs):
        cost = -self.ELBO(X=inputs)
        return cost

    def posterior_collapse_rate(self, X, eps, delta=0.01):
        batch_size = X.shape[0]
        with torch.no_grad():
            z_mean, z_log_sigma_sq = self.encoder(X)
            z_sigma_sq = torch.exp(z_log_sigma_sq)
            KL_element_wise = 0.5 * (
                    z_sigma_sq
                    - z_log_sigma_sq
                    + z_mean ** 2
                    - 1
            )
            z_posterior_collapse = torch.where(KL_element_wise < eps, 1.0, 0)
            prob_x_pc = torch.sum(z_posterior_collapse, dim=0) / batch_size
            z_collapse = torch.where(prob_x_pc >= 1 - delta, 1.0, 0)
            collapsed_rate = torch.sum(z_collapse) / self.z_dim
        return collapsed_rate.item()
