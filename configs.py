class ModelConfigs:
    def __init__(self):
        self.train_sigma = True
        self.init_sigma = 1
        self.latent_dim = 200
        self.data_dim = 784
        self.data_size = 1000
        self.explicit_mode = True
        self.device = 'cuda:0'
        self.encoder_type = 'mlp'  # [deep_linear, mlp]
        self.decoder_type = 'mlp'  # [linear, mlp , conv]


class AlgConfigs:
    def __init__(self):
        self.learning_rate = 0.0005
        self.save_checkpoint_freq = 1000
        self.train_epochs = 50000
        self.normalize_data = True
        self.preprocess_data = False
        self.eps = 1e-6

class Configs:
    def __init__(self):
        self.model_configs = ModelConfigs()
        self.alg_configs = AlgConfigs()


configs = Configs()
