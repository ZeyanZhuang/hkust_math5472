import torch
from configs import *
from vae import LinearVAE
from ppca import PPCA
from data_loader import DataSampler
import numpy as np
import matplotlib.pyplot as plt

data_sampler = DataSampler(configs)
data = data_sampler.sample(1000)
inputs_data = torch.from_numpy(data.astype(np.float32)).to(configs.model_configs.device)
ppca = PPCA(configs)
latent_dim_ls = [50 * i for i in range(7)]
ppca_log_p_ls = []
final_log_lik = []

for l_dm in latent_dim_ls:
    ppca.set_latent_dim(l_dm)
    avg_log_lik = ppca.fit(data)
    ppca_log_p_ls.append(avg_log_lik)

for l_dm in latent_dim_ls:
    configs = Configs()
    configs.model_configs.latent_dim = l_dm
    vae = LinearVAE(configs)
    optim = torch.optim.Adam(vae.parameters(), lr=configs.alg_configs.learning_rate)
    loss = 0
    for i in range(configs.alg_configs.train_epochs):
        loss = vae.forward(inputs_data)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % 1000 == 0:
            print('latent space: {}, epoch: {}, train elbo: {}'.format(l_dm, i, -loss.item()))
    final_log_lik.append(-loss.item())

ppca_log_p_ls_fix_sigma = []
for l_dm in latent_dim_ls:
    ppca.set_latent_dim(l_dm)
    avg_log_lik = ppca.fit_fix_sigma(data, s_dim=50)
    ppca_log_p_ls_fix_sigma.append(avg_log_lik)


plt.figure(figsize=(8, 6))
plt.plot(latent_dim_ls, ppca_log_p_ls, c='g', linestyle='-', label='Exact likelihood')
plt.plot(latent_dim_ls, ppca_log_p_ls_fix_sigma, c='b', linestyle='-', label='Exact likelihood(MLE(50))')
plt.plot(latent_dim_ls, final_log_lik, c='r', linestyle='--', label='ELBO')

plt.xlabel('Hidden dimensions')
plt.title('Marginal log-likelihood of pPCA')
plt.grid(True)
plt.legend()
plt.savefig('./fig_2.jpg', dpi=500)
plt.show()