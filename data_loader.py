import numpy as np
import os
import gzip
import torch

def load_mnist(data_folder='./MNIST_data'):
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder, fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)


# (train_images, train_labels), (test_images, test_labels) = load_data('MNIST_data/')
class DataSampler:
    def __init__(self, configs):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_mnist()
        self.configs = configs

    def sample(self, data_size=1000):
        index = np.random.choice(np.arange(self.x_train.shape[0]), data_size)
        data = self.x_train[index, :]
        # data = self.x_train[:data_size, :]
        data = np.reshape(data.astype(np.float32), [-1, 784])
        if self.configs.alg_configs.preprocess_data:
            data = self.preprocess(data, eps=self.configs.alg_configs.eps)
            return data
        elif self.configs.alg_configs.normalize_data:
            data /= 255.0
        return data

    def sample_tensor(self, data_size=1000):
        sample = self.sample(data_size)
        sample = torch.from_numpy(sample).to(self.configs.model_configs.device)
        return sample

    def preprocess(self, X, eps):
        X = (X + np.random.rand(*X.shape)) / 256.0
        item = eps + (1 - 2 * eps) * X
        data = np.log(item / (1.0 - item))
        return data.astype(np.float32)