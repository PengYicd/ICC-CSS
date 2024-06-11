# Integrated Distributed Semantic Communication and Over-the-air Computation for Cooperative Spectrum Sensing  

**Peng Yi, Yang Cao, Xin Kang, and Ying-Chang Liang**

This is the implementation of the paper named [[2311.04791\] Integrated Distributed Semantic Communication and Over-the-air Computation for Cooperative Spectrum Sensing (arxiv.org)](https://arxiv.org/abs/2311.04791)

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
@article{yi2023integrated,
  title={Integrated Distributed Semantic Communication and Over-the-air Computation for Cooperative Spectrum Sensing},
  author={Peng Yi and
          Yang Cao and
          Xin Kang and
          Y.-C. Liang},
  journal={arXiv preprint arXiv:2311.04791},
  year={2023}
}
```
