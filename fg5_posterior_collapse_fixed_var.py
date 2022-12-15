import torch
from configs import *
from vae import LinearVAE, DeepVAE
from ppca import PPCA
from data_loader import DataSampler
import numpy as np

sigma_sq_list = [30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0001]
# sigma_sq_list = [0.1]
vaes = []

data_sampler = DataSampler(configs)
data = data_sampler.sample_tensor(1000)

# vae = LinearVAE(configs)
for sigma_sq in sigma_sq_list:
    vae_configs = Configs()
    vae_configs.model_configs.init_sigma = np.sqrt(sigma_sq)
    vae = DeepVAE(vae_configs)
    optim = torch.optim.Adam(vae.parameters(), lr=configs.alg_configs.learning_rate)
    for i in range(20000):
        loss = vae.forward(data)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % 1000 == 0:
            rate = vae.posterior_collapse_rate(data, 0.01, delta=0.01)
            print('epoch: {}, sigma square: {}, deep elbo: {}, collapse rate: {}'.format(i, sigma_sq, -loss.item(),
                                                                                        rate))
    vaes.append(vae)


eps_list = np.linspace(0, 1.7, 100)

vae = vaes[0]

ps_rate = [[] for i in range(11)]
for i, vae in enumerate(vaes):
    for eps in eps_list:
        rate = vae.posterior_collapse_rate(data, eps)
        ps_rate[i].append(rate)


## show results
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(14, 8))
fig.suptitle('Posterior collapse: MNIST (fixed variance)')

index = 0
for i in range(3):
    for j in range(4):
        if i == 2 and j == 3:
            break
        ax[i, j].plot(eps_list, ps_rate[index])
        ax[i, j].grid()
        ax[i, j].set_title(r'$\sigma^2$={}'.format(sigma_sq_list[index]))
        ax[i, j].set_xlabel(r'$\epsilon$')
        ax[i, j].set_ylabel(r'Collapse %')
        index += 1

fig.tight_layout()
plt.grid()

plt.savefig('fig_5.jpg', dpi=500)
plt.show()
