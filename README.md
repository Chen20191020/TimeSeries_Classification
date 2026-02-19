# Classification Method for Locally Stationary Time Series

We propose a numerically efficient, practically competitive, and theoretically rigorous classification method for distinguishing between two classes of locally stationary time series based on their time-domain, second-order characteristics. Our approach is built upon the structural approximation that any short-range dependent, locally stationary time series can be well-approximated by a time-varying autoregressive (AR) process. Using this foundation, we extract discriminative features from the corresponding time-varying AR coefficient functions, which capture key features. To achieve robust classification, we employ an ensemble aggregation step that amplifies even mild differences in autocovariance structures between classes, followed by a distance-based thresholding mechanism. A key advantage of our method is its interpretability; Through extensive numerical simulations and real EEG data, we demonstrate that this approach outperforms a variety of state-of-the-art solutions, including wavelet-based, tree-based, and modern deep learning methods. This repository contains the R and Python implementation of the classification method proposed. 

Our proposed method is described in the paper:
C. Qian, X. Ding, and L. Li. Structural Classification of Locally Stationary Time Series Based on Second-order Characteristics, arXiv: 2507.04237. https://arxiv.org/abs/2507.04237

# Content

Each folder is named after the corresponding table or figure in the paper. For example, the `Table1` folder contains the code used to generate Table 1, and the same convention applies to other tables and figures. Within each folder, code files are named according to the methods used in the paper. For instance, `Arsenal_demo.py` corresponds to the Arsenal method.

To distinguish the LSW method from LSW simulation models, we use the prefix `LSWmethod` throughout the repository to refer specifically to the LSW method implemented in the paper.

Our proposed method, along with DWT, LSW, and FLogistic, is implemented in R. The remaining methods—Arsenal, MultiRocket, Rocket, STSForest, Shapelet, TimeCNN, and TimeRNN—are implemented in Python.

**Folder Table 1**, **Folder Table S.1**, **Folder Table S.2**, **Folder Table S.3**, **Folder Table S.4**, include two subfolders, `Balanced` and `Unbalanced`. The `Balanced` folder contains code for all methods under the setting $N_1 = 100, N_2 = 100$, while the `Unbalanced` folder contains code for all methods under the setting $N_1 = 50, N_2 = 100$.


**Folder Table 1** contains the code for classification accuracy under noise distribution (i). It includes `Simulation_setup_eps1.R` and subfolders, `Balanced` and `Unbalanced`. 

**Folder Figure 2** contains the code for the Boxplot of extracting features. It includes `Figure2_draw.R` to draw the plot, and the code files for each method. 

**Folder Figure 3** contains the code for Figure 3(a), Figure 3(b), and Figure 3(c). It includes 3 subfolders, `Figure 3(a)`, `Figure 3(b)` and `Figure 3(c)`. Each subfolder contains the code files for each method and one file for drawing the plot.

**Folder Table 2** contains the code for classification accuracy for EEG data across all methods. It includes `DWT_realdata.R`, `FLogistic_realdata.R`, `LSWmethod_realdata.R`, `Proposed_realdata.R`, and `methods_inpy_realdata.py`. In addition, the code file `methods_inpy_realdata.py` contains the code for methods Arsenal, MultiRocket, Rocket, STSForest, Shapelet, TimeCNN, and TimeRNN. 

**Folder Figure 4** contains the code for Figure 4(a) and Figure 4(b). It includes 2 subfolders, `Figure 4(a)` and `Figure 4(b)`. `Figure 4(a)` contains code file `Figure_4(a).R` and `Figure 4(b)` contains `Figure_4(b)_draw.R` to draw the plot and the rest code files are same as in **Folder Table 2**. However, we use the output majority voting results.

**Folder Table S.1** contains the code for classification accuracy under noise distribution (ii). It includes `Simulation_setup_eps2.R` and subfolders, `Balanced` and `Unbalanced`. 

**Folder Table S.2** contains the code for classification accuracy under noise distribution (iii). It includes `Simulation_setup_eps3.R` and subfolders, `Balanced` and `Unbalanced`. 

**Folder Table S.3** contains the code for classification accuracy under LSW processes. It includes `SimulationLSW_setup.R` and subfolders, `Balanced` and `Unbalanced`. 

**Folder Table S.4** contains the code for classification accuracy when differences lie in the mean. It includes `Simulation_setup_meandiff.R` and subfolders, `Balanced` and `Unbalanced`. 

**Folder Table S.5** contains the code for computing classification accuracy under the change-point setup, both without modification and after modification. It includes `Setup_1.R`, `Setup_2.R` for the results without modification, and `Change_detect_Setup_1.R` and `Change_detect_Setup_2.R` for the results after modification.

**Folder Table S.6** contains the code for classification accuracy under selected models with various features. It includes `selected_model_setup.R` for simulation setup and `All_aggfeature.R`, `Half_aggfeature.R`, `L2Dis_aggfeature.R`, `Min_aggfeature.R`, `Minvary_aggfeature.R`, `Proposed_aggfeature.R` for aggregate feature all, first-half, L2, min, min-vary, and proposed respectively.

**Folder Table S.7** contains the code for classification accuracy with different training sample setups. It includes `Proposed_varyn1n2n3_demo.R`. 

**Folder Table S.8** contains the code for computational time (in seconds) for different methods. It includes `Arsenal_comtime.py`, `DWT_comtime.R`, `FLogistic_comtime.R`, `LSWmethod_comtime.R`, `MultiRocket_comtime.py`, `Proposed_comtime.R`, `Rocket_comtime.py`, `STSForest_comtime.py`, `Shapelet_comtime.py`, `TimeCNN_comtime.py`, `TimeRNN_comtime.py`. Each file computes the computational time for the C3 and CZ channels across frequencies from 1 Hz to 12 Hz for the corresponding method.

**Folder Table S.9** contains the code for classification accuracy under unit root and long memory models. It includes `model_ur_lm_setup.R` for simulation setup and `All_aggfeature.R`, `Half_aggfeature.R`, `L2Dis_aggfeature.R`, `Min_aggfeature.R`, `Minvary_aggfeature.R`, `Proposed_aggfeature.R` for aggregate feature all, first-half, L2, min, min-vary, and proposed respectively.

**Folder Figure S.1** contains the code for classification accuracy LSW and DWT when the time series length increases. It includes `FigureS.1_draw.R` to draw the plot, `DWT_n_vary.R` and `LSWmethod_n_vary.R` for simulation. 



# System Requirements

The method requires only a standard computer with enough RAM to support the operations defined by a user. For optimal performance, we recommend a computer with the following specs:

* **RAM:** 16+ GB
* **CPU:** 4+ cores, 3.3+ GHz/core

### Software Versions
The implementation is provided in both R and Python and has been tested under the following versions:
* **R:** Version 4.1.1 or higher
* **Python:** Version 3.8 or higher

## 📦 Installation & Usage

### 1. R Environment
Ensure you have the necessary R packages installed. If `FREG` cannot be installed directly, please use the provided `FREG` file for manual installation.
```r
install.packages(c("wavelets", "MASS", "sparsediscrim", "fda", "refund", "wavethresh", "FREG"))
```

### 1. Python Environment
Ensure you have the necessary Python packages installed.
```python
from sklearn.metrics import accuracy_score
from aeon.classification.deep_learning import TimeCNNClassifier
import pandas as pd
import numpy as np
from time import time
import time
from aeon.classification.convolution_based import (
    Arsenal,
    MultiRocketClassifier,
    RocketClassifier,
)
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from sklearn.ensemble import RandomForestClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier
import rpy2.robjects as robjects
from sklearn import metrics

from aeon.classification.interval_based import (
    SupervisedTimeSeriesForest,
)

```


# Notes on How to Run the Code

For all real-data experiments (EEG), noise terms have already been included. Please manually specify the data paths as needed when loading the data.

To run the proposed method, first execute:

- `Fast_Proposed_method.R`

## Simulation Tables

- **Table 1**:  
  First run `Simulation_setup_eps1.R`.

- **Table S.1**:  
  First run `Simulation_setup_eps2.R`.

- **Table S.2**:  
  First run `Simulation_setup_eps3.R`.

- **Table S.3**:  
  For Python-based methods, the simulation data can be downloaded from:  
  https://drive.google.com/drive/folders/1J1zWdC41zoJ_QWxubKw-0LUxeZavRYmu?usp=sharing  
  Please manually specify the file paths.

  For R-based methods, first run `SimulationLSW_setup.R`.

- **Table S.4**:  
  First run `Simulation_setup_meandiff.R`.

- **Table S.6**:  
  The data can be downloaded from:  
  https://drive.google.com/drive/folders/10wuTVadTLqrvo7aO8X79Zsm-uE5eGcX1?usp=sharing  
  Please manually specify the data paths.
  For simulation, first run `selected_model_setup.R`.

- **Table S.8**:  
  The data can be downloaded from:  
  https://drive.google.com/drive/folders/1ditsD4G6S8q77yTPiAdolQcC5U8FyHHo?usp=sharing
  Please manually specify the data paths.
- **Table S.9**:  
  First run `model_ur_lm_setup.R`.

## Real Data Results

- **Table 2 and Figure 4**:  
  The data can be downloaded from:  
  https://drive.google.com/drive/folders/10wuTVadTLqrvo7aO8X79Zsm-uE5eGcX1?usp=sharing  

  Please manually specify the data paths after downloading.

