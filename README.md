# ECCGNN
# Environmental Consistency-Constrained Graph Neural Networks for Landslide susceptibility evaluation

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

this repo contains code accompaning the paper, [ Environmental Consistency-Constrained Graph Neural Networks for Reliable Landslide Susceptibility Evaluation](). It contains the code of 1) Environmental classification; 2) Graph construction; 3) the GraphSAGE model for LSM prediction. Also, it provides the drawing of most experimental figures.

##Table of Contents

- [Background](#background)
- [Dependencies](#dependencies)
- [Data](#data)
- [Usage](#usage)
- [Contact](#contact)


## Background
Due to complex geoenvironmental settings, landslide susceptibility evaluation (LSE) is limited by heterogeneous, lopsided, and insufficient samples. Conventional methods without consideration of surrounding environments cannot effectively generalize the characteristics of landslides and have a high false alarm rate and missed alarm rate. Low-quality landslide inventories have become a challenge for statistical methods. This paper proposes an environmental consistency-constrained graph neural network to overcome this problem. We correlated geographic nodes in different locations to form a graph based on environmental consistency, which makes the distribution of landslide features more orderly. Then, a graph neural network (GNN) was used to aggregate node information in the graph and to achieve LSE. In detail, we adopted the terrain polygon approximation method to divide geographic units to improve the rationality of the boundary and designed a balanced sampling strategy to improve model performance.


## Dependencies

This code is implemented with the anaconda environment:
* python 3.7
* pytorch
* dgl
* gdal 3.0.4
* numpy 1.19.5
* pandas 0.25.2
* scipy 1.3.0
* scikit-learn 0.21.2
* sklearn 0.0
* xlrd

## Data

* The experimental data is compressed in file `./datasets/datasets.rar`, where `FJ_contents.csv` and `FL_contents.csv` contain the features of each node V in area FJ and FL separately, `FJ_edges.csv` and `FL_edges.csv` contain the information about edges E in both area.
* The `label` field represents different node labels, '1' means landslide samples, '4' means nonlandslide samples, Others represent unlabeled nodes.
* `FJ_contents_raw.txt` and `FL_contents_raw.txt` contain orginal geographic unit features.


## Usage

* For the environmental classification and graph construction, see `./datasets/LS_data_preprocessing.py` and pretrain the base model. 
* For the the GraphSAGE model for LSM prediction, see `./modules/sage_train_full.py`, it will predict the susceptibility for each node in `xx_contents.csv`



## Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/Sunshine96223/ECCGNN/issues).

