# Deep Fuzzy Cognitive Maps with Vector Inference  for Multivariate Time Series Long-Term Forecasting
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)

##  0.Corrections to Formulas
**⚠️ Please note that two symbol errors were found in the paper and have now been corrected. The correct versions of the formulas are as follows:** 

Equation (15) has been corrected to:
<p align="center">
<img src=".\pic\eq15.png" height = "360"   alt="" align=center />  
<br><br>
</p>
Equation (22) has been corrected to:
<p align="center">
<img src=".\pic\eq22.png" height = "360"   alt="" align=center />  
<br><br>
</p>
We apologize for any inconvenience this may have caused. Thank you for your understanding and support!


##  1.Program Overview
(1) First, multidimensional Gaussian information granules are used to granulate the data in the training set, thereby obtaining the optimal segmentation method. For specific details, please refer to the literature “_Design Gaussian information granule based on the principle of justifiable granularity: A multi-dimensional perspective._” (Since this part of the method is adapted from existing work, I have reproduced the approach accordingly. The code is not publicly available at present, but can be provided upon request.)

(2) Next, run the “main_fcm” file in the util folder. Commands for training *VI-DFCM*  on the SIEC dataset:
  ```bash
   # SIEC
  --data ./steel --grain_seq_len 32 --grain_step_start 24 --grain_step 8
  ```

The key parameters are introduced as follows:
  `grain_seq_len` represents the length of each information granule, `grain_step_start` denotes the non-overlapping length of each granule and also serves as the prediction length for each iteration, while `grain_step` is the overlapping   length between neighboring granules. The values of `grain_seq_len` and `grain_step` are obtained from the segmentation results in the first step.

  Among these parameters, `grain_step_start` is a hyperparameter, and you only need to select an appropriate value based on the input and output length requirements of the task. For example, if the maximum input sequence length is 50 and the output length is 168, then 24 is chosen for two reasons: (1) this approach provides sufficient space for searching the optimal sliding window length, and the information granule length can be optimized within the range of 25 to 50; (2) the output length of 168 is an integer multiple of 24.




## 2.Paper Overview
<p align="center">
<img src=".\pic\1The overall architecture.png" height = "360"   alt="" align=center />  
<br><br>
<b>Figure 1.</b> The overall architecture.
</p>


Figure 1 shows the overall training process of VI-DFCM. VI-DFCM achieves long-term prediction through the synergistic effect of multiple iterations by the VI-FCM submodel and a single iteration by the Cross VI-FCM submodel.

### (1).Data Preprocessing
<p align="center">
<img src=".\pic\2Data Preprocessing.png" height = "360" width="50%" alt="" align=center />
<br><br>
<b>Figure 1.</b> Data preprocessing.
</p>


In our designed experiments, the prediction lengths are {168, 720}, and the input sequence lengths for each baseline model are {50, 100}, respectively. Therefore, when performing information granulation, we require that the length of the obtained optimal information granule does not exceed the input sequence length of each baseline, i.e., the input length of VI-DFCM should be less than or equal to {50, 100}.
### (2).VI-FCM submodel
<p align="center">
<img src=".\pic\3VI-FCM.png" height = "360" width="50%" alt="" align=center />
<br><br>
<b>Figure 1.</b> VI-FCM submodel.
</p>

The inference process of the VI-FCM submodel. The transfer function of VI-FCM is given by:

$A_i(S+1) = \frac{1}{1 + e^{-\lambda \left( \sum_{j=1}^{N} \left(\gamma_{i,j}  w_{i,j,:} \right) \circ A_j(S) + u_i(S) \right)}}$

where $\circ$ denotes the Hadamard product, $\lambda$ represents the steepness parameter of the transfer function near zero, and $\boldsymbol{w}\in\mathbb{R}^{N\times N \times I}$ is the weight tensor.

🔎 Equations `(12)–(15)` correspond to code in `models/VIDFCM.py`, lines `25–54`.
### (3).Cross VI-FCM submodel
<p align="center">
<img src=".\pic\4Cross VI FCM.png" height = "360" width="50%" alt="" align=center />
<br><br>
<b>Figure 1.</b> Cross VI-FCM submodel.
</p>

The inference process of the Cross VI-FCM submodel. The transfer function of Cross VI-FCM is given by:

$A_i(S+1) = \frac{1}{1 + e^{-\lambda \left( \sum_{j=1}^{N}  \left(\gamma_{i,j}^{CF}  w_{i,j,:}\right) \circ A_j(S) + u_i(S) \right)}}$

🔎 Equation `(18)` correspond to code in `models/VIFCM_Cross.py`, lines `31`.

🔎 Equations `(19)–(22)` correspond to code in `models/VIDFCM.py`, lines `68–90`.
### (4). EXPERIMENTAL RESULTS

<p align="center">
<img src=".\pic\5experiment.png" height = "360" width="70%" alt="" align=center />
<br><br>
<b>Table 1.</b> Comparison of MSE  Between VI-DFCM and Other Baseline Models..
</p>

***Baseline models:***

- LSTM-Informer： Li, Jiang-Cheng, et al. "Enhancing financial time series forecasting with hybrid Deep Learning: CEEMDAN-Informer-LSTM model." _Applied Soft Computing_ (2025): 113241.

- CNN-Transformer: Lou, Benxiao, et al. "Multi-source data-driven short-term remaining driving range prediction for electric vehicles: A hybrid CNN-transformer framework." _Energy_ (2025): 137564.

- CFCM: Ouyang, Chenxi, et al. "Constructing spatial relationship and temporal relationship oriented composite fuzzy cognitive maps for multivariate time series forecasting." _IEEE Transactions on Fuzzy Systems_ 32.8 (2024): 4338-4351.

- DAFCM: Qin D, Peng Z, Wu L. Deep attention fuzzy cognitive maps for interpretable multivariate time series prediction[J]. _Knowledge-Based Systems_, 2023, 275: 110700.

- DFCM: Wang J, Peng Z, Wang X, et al. Deep fuzzy cognitive maps for interpretable multivariate time series prediction[J]. _IEEE transactions on fuzzy systems_, 2020, 29(9): 2647-2660.

- GA-FCM

- SARIMA

## 3.Requirements

- Python 3.9
- numpy ==  1.26.3
- pandas == 2.2.3
- torch ==  2.6.0
- tqdm == 4.67.1




## 4.Data
1. [ETTTh1,ETTm1](https://github.com/zhouhaoyi/ETDataset)
2. [PCTC](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city)
3. [SIEC](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption)
4. [IHEPC](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power)
5. [BSD](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
6. [SYAQ](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)
7. [WTH](https://www.ncei.noaa.gov/data/local-climatological-data/)
