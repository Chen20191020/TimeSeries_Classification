# Classification Method for Locally Stationary Time Series

We propose a numerically efficient, practically competitive, and theoretically rigorous classification method for distinguishing between two classes of locally stationary time series based on their time-domain, second-order characteristics. Our approach is built upon the structural approximation that any short-range dependent, locally stationary time series can be well-approximated by a time-varying autoregressive (AR) process. Using this foundation, we extract discriminative features from the corresponding time-varying AR coefficient functions, which capture key features. To achieve robust classification, we employ an ensemble aggregation step that amplifies even mild differences in autocovariance structures between classes, followed by a distance-based thresholding mechanism. A key advantage of our method is its interpretability; Through extensive numerical simulations and real EEG data, we demonstrate that this approach outperforms a variety of state-of-the-art solutions, including wavelet-based, tree-based, and modern deep learning methods. This repository contains the R and Python implementation of the classification method proposed. 

Our proposed method is described in the paper:
C. Qian, X. Ding, and L. Li. Structural Classification of Locally Stationary Time Series Based on Second-order Characteristics, arXiv: 2507.04237. https://arxiv.org/abs/2507.04237

# Content

Each folder is named after the corresponding table or figure in the paper. For example, the folder Table1 contains the code used to generate Table 1, and similarly for other tables and figures. Within each folder, code files are named after the methods used in the paper. For instance, Arsenal_demo.py corresponds to the Arsenal method. 

To distinguish between the LSW method and LSW simulation models, we use the prefix LSWmethod throughout the repository to refer specifically to the LSW method implemented in the paper.

Our proposed method, along with DWT, LSW, and FLogistic, is implemented in R. The remaining methods—Arsenal, MultiRocket, Rocket, STSForest, Shapelet, TimeCNN, and TimeRNN—are implemented in Python.



# System Requirements

# How to use 
To run the proposed method, first run file Fast_Proposed_method.R. In order to run FLogistic method, first install the package "FREG" using FREG file. For Table 1, first run the file Simulation_setup_eps1.R. For Table S.1, first run the file Simulation_setup_eps2.R. For Table S.2, first run the file Simulation_setup_eps3.R. 

For Table S.3 The Simulation Data used for methods in Python can be download from https://drive.google.com/drive/folders/1J1zWdC41zoJ_QWxubKw-0LUxeZavRYmu?usp=sharing, the path of file should manually input in. For the method in R, first run file SimulationLSW_setup.R. 

For Table S.4 first run the file Simulation_setup_meandiff.R.  For Table S.8 the Data can be download from https://drive.google.com/drive/folders/10wuTVadTLqrvo7aO8X79Zsm-uE5eGcX1?usp=sharing

