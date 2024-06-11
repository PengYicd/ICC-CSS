import torch
from torch.utils.data import DataLoader
from PU_data import PU_data_dataset
from DSCmodel import DistributiedSC
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from config import get_config


if __name__ == '__main__':
    args = get_config()
    device = torch.device(args.device)
    torch.set_default_tensor_type(torch.FloatTensor)

    total_size = 51200

    net = DistributiedSC(args)
    net.load_state_dict(torch.load("models/best.pt"))
    net.to(device).eval()
    

    P_f_list = np.arange(-30, 0, step=1) / 10 
    P_f_list = list(np.power(10, P_f_list))
    P_d_list = list()

    my_test_1 = PU_data_dataset(args, total_size, 'test_1', False)
    my_loder_test_1 = DataLoader(dataset=my_test_1, batch_size=256, shuffle=False)
    my_test_0 = PU_data_dataset(args, total_size, 'test_0', False)
    my_loder_test_0 = DataLoader(dataset=my_test_0, batch_size=256, shuffle=False)
    gama_list = list()

    pbar = tqdm(my_loder_test_0)
    P_d_P_f_list = list()
    for x, y in pbar:
        x = x.to(torch.float32).to(device)
        y = y.to(torch.float32).to(device)
        hat_y = net(x)
        hat_y = torch.sigmoid(hat_y)
        m = hat_y.detach().cpu().numpy()
        P_d_P_f_list.append(list(m[:,0]))
    P_d_P_f_list = np.array(P_d_P_f_list).flatten()
    P_d_P_f_list.sort()

    for i in P_f_list:
        gama_list.append(P_d_P_f_list[round(total_size * (1-i))])
    print("gama_list:")
    print(gama_list)

    pbar = tqdm(my_loder_test_1)
    P_d_P_f_list = list()
    for x, y in pbar:
        x = x.to(torch.float32).to(device)
        y = y.to(torch.float32).to(device)
        hat_y = net(x) 
        hat_y = torch.sigmoid(hat_y)
        m = hat_y.detach().cpu().numpy()
        P_d_P_f_list.append(list(m[:,0]))
    P_d_P_f_list = np.array(P_d_P_f_list).flatten()
    for i in gama_list:
        P_d_list.append( sum( P_d_P_f_list > i) / total_size ) 

    
    P_f_list.append(1)
    P_d_list.append(1)
    print(P_f_list)
    print(P_d_list)
    print(f'Pf=0.1 : {P_d_list[-11]}' )

    plt.plot(P_f_list, P_d_list)
    plt.xscale('log')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.axis(xmin=10**(-3), xmax=10**(0), ymin=0.0, ymax=1.0)
    title = 'ICC-CSS' + ' K:' + str(args.num_sensor) + ' M:' + str(args.dimension) + ' N:'+ str(args.samples) + 'sensing SNR:' + str(args.SNR_sensing) + 'dB' + 'report SNR:' + str(args.SNR_reporting) + 'dB'
    plt.title(title)
    plt.xlabel('Probability of False Alarm', fontproperties='Arial', fontweight='bold')
    plt.ylabel('Probalitity of Detection', fontproperties='Arial', fontweight='bold')
    plt.grid(visible=True, which='both', axis='both')
    plt.savefig('models/ROC.jpg') 
    plt.show() 


