import torch
from configs import *
from vae import LinearVAE
from ppca import PPCA
from data_loader import DataSampler
import numpy as np


data_sampler = DataSampler(configs)
data = data_sampler.sample(1000)
inputs_data = torch.from_numpy(data.astype(np.float32)).to(configs.model_configs.device)

configs = Configs() # default
configs_sto = Configs()
configs_sto.model_configs.explicit_mode = False

ppca = PPCA(configs)
avg_log_lik = ppca.fit(data)
print(avg_log_lik)

vae_ana = LinearVAE(configs)
vae_sto = LinearVAE(configs_sto)

## train stochastic model
optim_sto = torch.optim.Adam(vae_sto.parameters(), lr=configs.alg_configs.learning_rate)
vae_elbo_sto_list = []

## train analytic model
optim_ana = torch.optim.Adam(vae_ana.parameters(), lr=configs.alg_configs.learning_rate)
vae_elbo_ana_list = []

for i in range(configs.alg_configs.train_epochs):
    loss_sto = vae_sto.forward(inputs_data)
    optim_sto.zero_grad()
    loss_sto.backward()
    optim_sto.step()
    elbo_sto = -loss_sto.item()
    vae_elbo_sto_list.append(elbo_sto)

    loss_ana = vae_ana.forward(inputs_data)
    optim_ana.zero_grad()
    loss_ana.backward()
    optim_ana.step()
    elbo_ana = -loss_ana.item()
    vae_elbo_ana_list.append(elbo_ana)
    if i % 200 == 0:
        print('stochastic model, epoch: {}, log likelihood: {}, analytic elbo: {}, stochastic elbo: {}'.format(i, avg_log_lik, elbo_ana, elbo_sto))




import matplotlib.pyplot as plt
epochs = [i for i in range(len(vae_elbo_sto_list))]
plt.plot(epochs, vae_elbo_ana_list, c='b', linestyle='-', label='Analytic')
plt.plot(epochs, vae_elbo_sto_list, c='orange', linestyle='-', label='Stochastic')
plt.axhline(y=avg_log_lik, c='r', linestyle='-.', label='pPCA MLE')
plt.grid()
plt.xlabel('training epochs')
plt.ylabel('ELBO')
plt.title('Training loss for stochastic vs. analytic ELBO')
plt.legend()
plt.savefig('fig_3.jpg', dpi=500)
plt.show()
