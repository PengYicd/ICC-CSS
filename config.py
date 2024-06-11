import argparse
import numpy as np


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='random seed to generate data')
    parser.add_argument('--dimension', default=28, type=int, help='the number of antanas for each sensor')
    parser.add_argument('--samples', default=100, type=int, help='the number of samples in each samping period')
    parser.add_argument('--num_sensor', default=6, type=int, help='number of sensors')
    parser.add_argument('--num_PU', default=1, type=int, help='number of the moving devices')
    parser.add_argument("--coe_antenna", type=float, default=0.75, help="by default 0.75. the correlation coefficient of antennas")

    # Deep learning settings
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--training_epoch', default=300, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)

    # Channel settings
    parser.add_argument("--channel_type", type=str, default='Rician', help="by default Rician, trained under Rician.  AWGN Rician Rayleigh")
    parser.add_argument("--k_factor", type=float, default=0, help="by default 0, trained under 0 dB. K factor for Rician channel, dB")
    parser.add_argument("--iota", type=float, default=0.9, help="by default 0.9, trained under 0.9. channel estimation uncertainty")
    
    parser.add_argument("--SNR_sensing", type=float, default=-15, help="by default -15, trained under -15 dB. SNR in reporting channel, dB")
    parser.add_argument("--SNR_reporting", type=float, default=0, help="by default 0, trained under -10 dB. SNR in reporting channel, dB")

    FLAGS, _ = parser.parse_known_args()

    return FLAGS

