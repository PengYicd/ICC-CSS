import numpy as np
import torch, time, random
from torch.utils.data import Dataset, DataLoader
from math import log10, pi
import matplotlib.pyplot as plt
from config import get_config
from tqdm import trange, tqdm


class PU_data_dataset(Dataset):
    def __init__(self, args, total_size, stage, use_store) -> None:
        super().__init__()
        self.total_size = total_size
        self.stage = stage
        self.args = args
        self.SNR_sensing = args.SNR_sensing
        a = np.arange(self.args.dimension)
        self.R_h = self.args.coe_antenna ** np.abs(a-a[:, np.newaxis])
        self.use_store = use_store
        if use_store:
            if self.stage == 'train':
                self.data_x = np.load('data_x.npy')
                self.data_y = np.load('data_y.npy') 
            

    def __getitem__(self, index):
        if self.use_store:
            return self.data_x[index, :, :, :, :], self.data_y[index, :]
        else:
            self.snr_linear = 10**(self.SNR_sensing/10)
            if self.stage == 'train':
                self.flag = np.random.randint(2, size=1) 
            elif self.stage == 'test_0':
                self.flag = np.zeros(1)
            elif self.stage == 'test_1':
                self.flag = np.ones(1)
            return self.PU_data_set() 
        # return self.data_x[index, :, :, :, :], self.data_y[index, :]

    def __len__(self):
        return self.total_size

    #  circularly symmetric complex Gaussian (CSCG) 
    def cscg(self):
        x = np.random.randn(self.args.dimension, 1) + np.random.randn(self.args.dimension, 1) * 1.j
        return x / np.sqrt(2)

    def sample_from_CM(self, R):
        xn = np.random.multivariate_normal(mean=np.zeros(self.args.dimension), cov=R, size=self.args.num_sensor) + \
            np.random.multivariate_normal(mean=np.zeros(self.args.dimension), cov=R, size=self.args.num_sensor) * 1.j
        return xn / np.sqrt(2)

    def PU_data_set(self):
        x_temp = self.PU_data_rec_N_sample(self.snr_linear)
        x = np.array([[np.real(x_temp[i, :, :]), np.imag(x_temp[i, :, :])] for i in range(self.args.num_sensor)]) 
        # x_avg = np.mean(x, axis=(2, 3), keepdims=True)
        # x_std = np.std(x, axis=(2, 3), keepdims=True)
        # x_normalized = (x - x_avg) / x_std
        if self.flag == 1:
            y = np.array([1])
        elif self.flag == 0:
            y = np.array([0])
        # plt.matshow(x[0])
        # plt.show()
        return x, y

    # PU signal R_s
    def PU_data_rec_N_sample(self, snr_linear):
        h_m = self.sample_from_CM(self.R_h) # (num_sensor, num_dimension)
        h_m = h_m[:,:,np.newaxis]
        if self.flag == 1:
            s_n = (np.random.randn(self.args.samples, 1) + np.random.randn(self.args.samples, 1) * 1.j) / np.sqrt(2) * np.sqrt(snr_linear)
            u_n = (np.random.randn(self.args.num_sensor, self.args.dimension, self.args.samples) + np.random.randn(self.args.num_sensor, self.args.dimension, self.args.samples) * 1.j) / np.sqrt(2)
            x_n = h_m @ s_n.T + u_n
        elif self.flag == 0:
            u_n = (np.random.randn(self.args.num_sensor, self.args.dimension, self.args.samples) + np.random.randn(self.args.num_sensor, self.args.dimension, self.args.samples) * 1.j) / np.sqrt(2)
            x_n = u_n
        x_n_expand = np.expand_dims(x_n.swapaxes(1, 2), 3)
        R_x = x_n_expand @ x_n_expand.swapaxes(2, 3).conj()
        R_x_s = np.mean(R_x, axis=1)
        # plt.matshow(np.real(R_x_s[0,:,:]))
        # plt.show()
        return R_x_s


if __name__ == '__main__':
    # Initialize the dataset to accelerate training
    data_size = 25600
    args = get_config()
    data_x = np.empty((data_size, args.num_sensor, 2, 28, 28), dtype=np.float32)
    data_y = np.empty((data_size, 1), dtype=np.float32)
    start = time.time()
    data = PU_data_dataset(args, data_size, 'train', False)
    dataloader = DataLoader(dataset=data, batch_size=512)
    print(args.samples)
    pbar = tqdm(dataloader) 
    for i, (x, y) in enumerate(pbar):
        data_x[i*512: (i+1)*512, :, :, :, :] = x
        data_y[i*512: (i+1)*512, :] = y

    np.save('data_x.npy', data_x)
    np.save('data_y.npy', data_y)
   
    print(time.time() - start)



