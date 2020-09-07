# Graph regularized Restricted Boltzmann Machine (GraphRBM) for Representation Learning
This is the Matlab implementation of IEEE TNNLS'18 paper

[Graph regularized Restricted Boltzmann Machine](https://ieeexplore.ieee.org/abstract/document/7927417).

By [Dongdong Chen](https://scholar.google.co.uk/citations?user=eIrcIl8AAAAJ&hl=en), Jiancheng Lv, Zhang Yi.

The College of Computer Science, Sichuan University.

### Table of Contents
0. [Keywords](#Keywords)
0. [Abstract](#Abstract)
0. [Requirement](#Requirement)
0. [Usage](#Usage)
0. [Citation](#citation)

### Keywords

Restrcited Boltzmann Machine (RBM), Graph regularization, Manifold learning, Sparse representation, Structure preservation.

### Abstract

The restricted Boltzmann machine (RBM) has received an increasing amount of interest in recent years. It determines good mapping weights that capture useful latent features in an unsupervised manner. The RBM and its generalizations have been successfully applied to a variety of image classification and speech recognition tasks. However, most of the existing RBM-based models disregard the preservation of the data manifold structure. In many real applications, the data generally reside on a low-dimensional manifold embedded in high-dimensional ambient space. In this brief, we propose a novel graph regularized RBM to capture features and learning representations, explicitly considering the local manifold structure of the data. By imposing manifold-based locality that preserves constraints on the hidden layer of the RBM, the model ultimately learns sparse and discriminative representations. The representations can reflect data distributions while simultaneously preserving the local manifold structure of data. We test our model using several benchmark image data sets for unsupervised clustering and supervised classification problem. The results demonstrate that the performance of our method exceeds the state-of-the-art alternatives.

### Requirement
0. Matlab (>=2011)
0. download the datasets (COIL20, MNIST, etc.)
0. the code is for computation on GPU
0. a pre-defined graph matrix 'phi' (n x n) is required for all the experiments/datasets. Note if one plays with limited RAM, we suggest computing the neighbourhood graph (vector) progressively, i.e., init each column of 'phi' associated to each sample and ultimately get the whole 'phi' (matrix).

### Usage
0. run [main.m](https://github.com/edongdongchen/GraphRBM/blob/master/main.m) to start a simple example.
0. run [train_GraphRBM_bin_bin.m](https://github.com/edongdongchen/GraphRBM/blob/master/train_GraphRBM_bin_bin.m) to train a GraphRBM (binary to binary).


### Citation

If you find the codes are useful or used the codes in your research, it is appreciated if you cite:

    @article{chen2018graph,
    title={Graph regularized restricted Boltzmann machine},
    author={Chen, Dongdong and Lv, Jiancheng and Yi, Zhang},
    journal={IEEE transactions on neural networks and learning systems},
    volume={29},
    number={6},
    pages={2651--2659},
    year={2018},
    publisher={IEEE}
    }
