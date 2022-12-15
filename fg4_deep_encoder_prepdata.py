import torch
from configs import *
from vae import LinearVAE, DeepVAE
from ppca import PPCA
from data_loader import DataSampler


data_sampler = DataSampler(configs)
data = data_sampler.sample_tensor(1000)
ppca = PPCA(configs)
avg_log_lik = ppca.fit(data.cpu().numpy())
print(avg_log_lik)

configs_linear = Configs()
configs_linear.model_configs.explicit_mode = False
vae_l = LinearVAE(configs_linear)
optim_l = torch.optim.Adam(vae_l.parameters(), lr=configs.alg_configs.learning_rate)

configs_mlp = Configs()
vae_mlp = DeepVAE(configs_mlp)
optim_mlp = torch.optim.Adam(vae_mlp.parameters(), lr=configs.alg_configs.learning_rate)


configs_deep_l = Configs()
configs_mlp.model_configs.encoder_type = 'deep_linear'
vae_deep_l = DeepVAE(configs_deep_l)
optim_deep_l = torch.optim.Adam(vae_deep_l.parameters(), lr=configs.alg_configs.learning_rate)


l_elbo = []
deep_l_elbo = []
mlp_elbo = []

for i in range(10000):
    loss_l = vae_l.forward(data)
    loss_deep_l = vae_deep_l.forward(data)
    loss_mlp = vae_mlp.forward(data)

    optim_l.zero_grad()
    optim_deep_l.zero_grad()
    optim_mlp.zero_grad()

    loss_l.backward()
    loss_deep_l.backward()
    loss_mlp.backward()

    optim_l.step()
    optim_deep_l.step()
    optim_mlp.step()

    l_elbo.append(-loss_l.item())
    deep_l_elbo.append(-loss_deep_l.item())
    mlp_elbo.append(-loss_mlp.item())

    if i % 500 == 0:
        print('epoch: {}, init log px: {}, linear elbo: {}, deep linear elbo: {},  deep mlp elbo: {}'.format(
            i, avg_log_lik, -loss_l.item(), -loss_deep_l.item(), -loss_mlp.item()))


## print part
import matplotlib.pyplot as plt
epochs = [i for i in range(40000)]

beg_index = 0

plt.plot(epochs[beg_index:], l_elbo[beg_index:], c='b', linestyle='-', label='Linear VAE')
plt.plot(epochs[beg_index:], deep_l_elbo[beg_index:], c='g', linestyle='-', label='Deep Linear Encoder')
plt.plot(epochs[beg_index:], mlp_elbo[beg_index:], c='orange', linestyle='-', label='Nonlinear Encoder')


plt.axhline(y=avg_log_lik, c='r', linestyle='-.', label='pPCA MLE')
plt.grid()
plt.xlabel('training epochs')
plt.ylabel('ELBO')
plt.title('Linear Decoder VAE with varying encoders')
plt.legend()
plt.ylim((-3000, -1080))
plt.savefig('fig_4.jpg', dpi=500)
plt.show()