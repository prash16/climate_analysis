# Climate Analysis 


This data was constructed using LLNL's UQ Pipeline, was created under the auspices of the US Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344, was funded by LLNL's Uncertainty Quantification Strategic Initiative Laboratory Directed Research and Development Project under tracking code 10-SI-013, and is released under UCRL number LLNL-MISC-633994.



Data Set Information:

This dataset contains records of simulation crashes encountered during climate model uncertainty quantification (UQ) ensembles. 

Ensemble members were constructed using a Latin hypercube method in LLNL's UQ Pipeline software system to sample the uncertainties of 18 model parameters within the Parallel Ocean Program (POP2) component of the Community Climate System Model (CCSM4). 

Three separate Latin hypercube ensembles were conducted, each containing 180 ensemble members. 46 out of the 540 simulations failed for numerical reasons at combinations of parameter values. 

The goal is to use classification to predict simulation outcomes (fail or succeed) from input parameter values, and to use sensitivity analysis and feature selection to determine the causes of simulation crashes. 

[UCI Data Source](https://archive.ics.uci.edu/ml/datasets/climate+model+simulation+crashes)

Dependencies: - 

sklearn
pandas
numpy
matplotlib


Instructions : 

To run analysis  : 

Be in the home directory and run below scripts. 

Step1 : Download the  data using this script : [script](./src/download.ipynb)

Step 2 : Run descriptive stats using script :[descriptive](descriptive.ipynb)

Step 3 : Visualization : [viz](visualization.ipynb)

Step 4 : Run analysis  : [Feature and model selection](analysis.ipynb)

Step 5 : Analysis Report :[Analysis-pdf](./reports/re.pdf)  and [Analysis-md file ](./reports/report.md) 


Thank you for your patience !!!!

 

![](random.jpg)





