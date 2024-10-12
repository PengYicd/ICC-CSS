# Integrated Distributed Semantic Communication and Over-the-air Computation for Cooperative Spectrum Sensing  

**Peng Yi, Yang Cao, Xin Kang, and Ying-Chang Liang**

This is the implementation of the paper named [Integrated Distributed Semantic Communication and Over-the-air Computation for Cooperative Spectrum Sensing](https://ieeexplore.ieee.org/document/10693603) published in  IEEE Transactions on Communications.

### Configure

Cooperative Spectrum Sensing configuration, deep learning settings and channel settings can be found in the "config.py".

### Dataset

The primary user (PU) data is generated according to the configuration in "config.py", and the specific process is detailed in "PU_data.py".

### Train

Please use "CM_CNN_train.py" to train the model. 

To facilitate training, it is recommended to run the dataset file, i.e., "PU_data.py" first to generate the PU signals and store them as '.npy' format files. 

Then run "CM_CNN_train.py" with the change 

`mydata = PU_data_dataset(args, total_size, 'train', True)` 

### Test

Please use “CM_CNN_test.py” to test the model. 

### BibTex

```latex
@ARTICLE{10693603,
  author={Yi, Peng and Cao, Yang and Kang, Xin and Liang, Ying-Chang},
  journal={IEEE Transactions on Communications}, 
  title={Integrated Distributed Semantic Communication and Over-the-air Computation for Cooperative Spectrum Sensing}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Sensors;Semantics;Wireless sensor networks;Scalability;Wireless networks;Internet of Things;Feature extraction;Cooperative spectrum sensing;distributed semantic communication;over-the-air computation},
  doi={10.1109/TCOMM.2024.3468215}}

```
