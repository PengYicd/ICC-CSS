import torch, time, os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from PU_data import PU_data_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from DSCmodel import DistributiedSC
from config import get_config


def seed_everything(seed: int): # fix the seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = get_config()
    # seed_everything(args.seed)
    device = torch.device(args.device)
    torch.set_default_tensor_type(torch.FloatTensor)
    total_size = 25600

    mydata = PU_data_dataset(args, total_size, 'train', False)
    my_loder = DataLoader(dataset=mydata, batch_size=args.batch_size, shuffle=True)
    net = DistributiedSC(args)
    net.float().to(device).train()

    best_loss = 1
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean') 
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    loss_epoch = list()
    for i in range(args.training_epoch):
        train_loss = list()
        pbar = tqdm(my_loder)
        for j, (x, y) in enumerate(pbar):
            x = x.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)
            optimizer.zero_grad()
            hat_y = net(x)
            loss = criterion(hat_y, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        print("Epoch:{}, Loss:{:10f}, lr:{:10f}".format(i, np.average(np.array(train_loss)), optimizer.param_groups[-1]['lr']))
        loss_epoch.append(np.average(np.array(train_loss)))
        if loss_epoch[-1] < best_loss:
            torch.save(net.state_dict(), 'models/best.pt')
            best_loss = loss_epoch[-1]

    torch.save(net.state_dict(), 'models/final.pt') 
    plt.figure(dpi=800)
    plt.plot(loss_epoch)
    plt.xlabel('Epochs', fontproperties='Arial', fontweight='bold', fontsize=16)
    plt.ylabel('Traing Loss', fontproperties='Arial', fontweight='bold', fontsize=16)
    plt.grid(visible=True, which='both', axis='both', linestyle=':')
    plt.savefig('models/train_loss.jpg') 
