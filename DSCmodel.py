# -*- coding: utf-8 -*-
# import einops
import torch.nn as nn
import torch
import torch.nn.functional as F
from config import get_config
from thop import profile
from time import time
from math import sqrt
from einops import rearrange
import random


class DepthwiseSeparable(nn.Module):
    def __init__(self, C_in, C_out, kernel_size):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels=C_in, out_channels=C_in, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, groups=C_in)
        self.pw_conv = nn.Conv2d(in_channels=C_out, out_channels=C_out, kernel_size=1, stride=1, groups=1)
        self.act = nn.ELU()

    def forward(self, x):
        x1 = self.dw_conv(x)
        x2 = self.pw_conv(x1)
        x3 = self.act(x2)
        return x3 


class Inception(nn.Module):
    def __init__(self, C_in, C_out) -> None:
        super().__init__()
        self.dsc1 = DepthwiseSeparable(C_in, C_in, 3)
        self.dsc2 = DepthwiseSeparable(C_in, C_in, 5)
        self.dsc3 = DepthwiseSeparable(C_in, C_in, 7)
        self.conv1 = nn.Conv2d(C_in*3, C_in, 3, 1, 1)
        self.act1 = nn.ELU()
        self.conv2 = nn.Conv2d(C_in, C_out, 3, 2, 1)
        self.norm = nn.BatchNorm2d(C_out)
        self.act2 = nn.ELU()

    def forward(self, x):
        x1_1 = self.dsc1(x)
        x1_2 = self.dsc2(x)
        x1_3 = self.dsc3(x)
        x1 = torch.cat([x1_1, x1_2, x1_3], dim=1)
        x2 = self.act1(self.conv1(x1))
        x3 = self.norm(self.act2(self.conv2(x2 + x)))
        return x3


class ResidualModule(nn.Module):
    def __init__(self, f_in, f_out) -> None:
        super().__init__()
        self.fc1 = nn.Linear(f_in, f_in)
        self.act1 = nn.ELU()
        self.fc2 = nn.Linear(f_in, f_in)
        self.act2 = nn.ELU()
        self.fc3 = nn.Linear(f_in, f_out)
        self.act3 = nn.ELU()
        self.norm = nn.BatchNorm1d(f_out)

    def forward(self, x):
        x1 = self.act1(self.fc1(x))
        x2 = self.act2(self.fc2(x1))
        x3 = self.norm(self.act3(self.fc3(x1 + x2)))
        return x3


class FusionCenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lc1 = nn.Sequential(ResidualModule(16, 32), ResidualModule(32, 64), ResidualModule(64, 128))
        self.lc3 = nn.Sequential(ResidualModule(128, 64), ResidualModule(64, 32), ResidualModule(32, 16))
        self.lc5 = nn.Linear(16, 1)

    def forward(self, x):
        x1 = self.lc1(x)
        x3 = self.lc3(x1)
        x5 = self.lc5(x3)
        return x5


class PowerNorm(nn.Module):
    def __init__(self, ):
        super(PowerNorm, self).__init__()
        self.register_buffer('moving_power', torch.ones(1))

    def power_norm(self, x, moving_power, momentum=0.9):
        if not self.training:
            x_hat = torch.div(x, torch.sqrt(2 * moving_power))
        else:
            x_square = torch.mul(x, x)
            power = torch.mean(x_square)
            x_hat = torch.div(x, torch.sqrt(2 * power))
            moving_power = momentum * moving_power + (1.0 - momentum) * power
        return x_hat, moving_power

    def forward(self, x):
        if self.moving_power.device != x.device:
            self.moving_power = self.moving_power.to(x.device)
        y, self.moving_power = self.power_norm(x, self.moving_power, momentum=0.9)
        return y
    

class DistributiedSC(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.Sensor = nn.Sequential(Inception(2, 4), Inception(4, 8), Inception(8, 16), nn.AvgPool2d(4, 1))
        self.FC = FusionCenter()
        self.Power = PowerNorm()
        self.k_factor = 10**(args.k_factor / 10) # 10dB
        self.iota = args.iota
        self.SNR_reporting = args.SNR_reporting
        # self.connet = nn.Linear(args.num_sensor * 8 * 2, 8 * 2)

    def channel(self, x):
        # x : (batch, dim)
        if self.args.channel_type == 'AWGN':
            h = torch.ones(([x.shape[0], 8]), dtype=torch.complex64, device=self.args.device)
        elif self.args.channel_type == 'Rayleigh':
            h = torch.randn(([x.shape[0], 8]), dtype=torch.complex64, device=self.args.device)
        elif self.args.channel_type == 'Rician':
            h_rayleigh = torch.randn(([x.shape[0], 8]), dtype=torch.complex64, device=self.args.device)
            h = sqrt(self.k_factor/(self.k_factor+1)) + sqrt(1/(self.k_factor+1)) * h_rayleigh
        else:
            raise RuntimeError('None defined channel')
        
        bar_h = (h - sqrt(1 - self.iota ** 2) * torch.randn_like(h, device=self.args.device)) / self.iota
        b = (h / bar_h)
        x_reshape = rearrange(x, 'b (m s) -> b s m', m=2)
        real = torch.mul(torch.real(b), x_reshape[:, :, 0]) - torch.mul(torch.imag(b), x_reshape[:, :, 1])
        imag = torch.mul(torch.real(b), x_reshape[:, :, 1]) + torch.mul(torch.imag(b), x_reshape[:, :, 0])
        x_hat = torch.concat((real, imag), dim=1)
        # error = torch.sum(torch.abs(x_hat - x))
        return x_hat

    def forward(self, x): # (batch, num_sensor, 2, 28, 28)
        for i in range(self.args.num_sensor):
            if i == 0:
                y = self.channel(self.Power(torch.squeeze(self.Sensor(x[:,i,:,:,:]))))
            else:
                y += self.channel(self.Power(torch.squeeze(self.Sensor(x[:,i,:,:,:]))))

        noise = torch.randn(y.shape, device=self.args.device) / (2**(1/2)) / sqrt(10**(self.SNR_reporting/10))
        # noise = torch.zeros(y.shape, device=self.args.device)
        y_hat = self.FC(torch.div(y + noise, self.args.num_sensor))

        return y_hat


class Sensor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.Sensor = nn.Sequential(Inception(2, 4), Inception(4, 8), Inception(8, 16), nn.AvgPool2d(4, 1))

    def forward(self, x):
        y = torch.squeeze(self.Sensor(x[:,:,:,:]))
        return y


def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    args = get_config()
    args.device = 'cuda:0'
    batch_size = 25600
    x = torch.rand(batch_size, 6, 2, 28, 28).to(args.device)
    model = DistributiedSC(args).to(args.device)
    
    s = time()
    y = model(x)
    e = time()
    tt = (e-s)/batch_size * 1000
    print('running time:' + str(tt) + 'ms')


    flops, params = profile(model, inputs=(x, ))
    print('Mflops:', flops/batch_size/1000000)
    print('params:', params)

