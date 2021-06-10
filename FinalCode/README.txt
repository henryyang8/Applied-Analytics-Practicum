Data Sources
* Data_with_Labels.xlsx: These are the files containing the ：labels； for a subset of hospitals from all over the US.
* rand_hcris_cy_hosp_a_2021_02_01.csv: This is the zipped full hospital cost report information system (HCRIS) data file having all the financial metrics for all hospitals across several years
* Medicare IPPS charge data: This dataset contains the utilization and charge data for providers participating in IPPS (from hereby referred to as ．IPPS hospitals・) for 2018
* Medicare OPPS charge data: This dataset contains the utilization and charge data for providers participating in OPPS (from hereby referred to as ．OPPS hospitals・) for 2018
* Quality Data: This data contains quality scores along with a bunch of other related metrics for all VBP program participating hospitals in the US.
* These data sources had 1500+ variables available for analysis
Code Run Book
Development Environment
Operating System
* Windows
Python Version
* Python 3.7.6
Libraries: re, pandas, numpy, pickle, sklearn, matplotlib.pyplot
* from __future__ import division
* import re
* import pandas as pd
* from pandas_profiling import ProfileReport
* import numpy as np
* import matplotlib.pyplot as plt
* import pickle
* import sklearn
* from sklearn import tree
* from sklearn.model_selection import train_test_split
* from sklearn.ensemble import GradientBoostingClassifier
* from sklearn.tree import DecisionTreeClassifier
* from sklearn.metrics import classification_report,confusion_matrix
* from sklearn import preprocessing
* from sklearn.model_selection import validation_curve, train_test_split
* from sklearn.linear_model import LogisticRegression
* from sklearn.feature_selection import SelectFromModel
* from sklearn.preprocessing import StandardScaler
* from sklearn.model_selection import validation_curve
* from sklearn.metrics import multilabel_confusion_matrix
* from sklearn.metrics import accuracy_score
* import seaborn as sns
Code is structured into 4 jupyter notebooks
Base folder contains the following files and folders as shown below:
1. Data Preparation: Data_Preparation_Script_1.ipynb
2. Modeling: Modeling_Script_2.ipynb
3. Scoring: Scoring_Script_3.ipynb
4. Evaluation: Evaluation_Script_4.ipynb
Create the following folder structure in the base folder to run the code:
* QoS_Output_Results/
o Partitioned_Models/
* unseen_scored_predictions/
o Unpartitioned_Models/
* unseen_scored_predictions/
* QoS_Pickled_Models/
o Partitioned_Models/
o Unpartitioned_Models/
Required raw data files:
InputDatasets/
1. Data_with_Labels_new.xlsx
2. rand_hcris_cy_hosp_a_2021_02_01.csv
3. MEDICARE_CHARGE_INPATIENT_DRGALL_DRG_BY STATE_FY2018.CSV
4. MEDICARE_CHARGE_INPATIENT_DRGALL_DRG_NATIONAL_FY2018.CSV
5. MEDICARE_PROVIDER_CHARGE_INPATIENT_DRGALL_FY2018.CSV
6. MUP_OHP_R20_P04_V10_D18_APC_Provider.xlsx
7. MUP_OHP_R20_P04_V10_D18_APC_Summary.xlsx

Parameter Configurations:
Following parameter need to be set in the notebooks to run different versions of modeling:
* Data_Preparation_Script_1.ipynb
o Training_rand = "2018"
o Scoring_rand = "2019"
* Modeling_Script_2.ipynb
o create_models_by_partition = True
o lasso_penalty = 0.1
o GBT_max_depth = 1
o GBT_n_estimators = 500
o data_file =? "rand_2018.csv"
o target_col = 'Class_5'
* Scoring_Script_3.ipynb
o score_models_by_partition = False
o data_file = "rand_2019.csv"
o target_col = 'Class_5'
* Evaluation_Script_4.ipynb
o evauluate_models_by_partition = True
Output Files:
Following output files are generated from the listed notebooks:
* Data_Preparation_Script_1.ipynb
o rand_2018.csv
o rand_2019.csv
o holdout_sample.csv
* Modeling_Script_2.ipynb
o QoS_Output_Results/Unpartitioned_Models/featureimportance_table.csv
o QoS_Output_Results/Unpartitioned_Models/predictions_table.csv
o QoS_Pickled_Models/Unpartitioned_Models/All_model.pkl
o QoS_Pickled_Models/Unpartitioned_Models/important_variables.pkl
o QoS_Output_Results/Partitioned_Models/featureimportance_table.csv
o QoS_Output_Results/Partitioned_Models/predictions_table.csv
o QoS_Pickled_Models/Partitioned_Models/1Star_model.pkl
o QoS_Pickled_Models/Partitioned_Models/2Star_model.pkl
o QoS_Pickled_Models/Partitioned_Models/3Star_model.pkl
o QoS_Pickled_Models/Partitioned_Models/important_variables.pkl
* Scoring_Script_3.ipynb
o QoS_Output_Results/Unpartitioned_Models/predictions_table.csv
o QoS_Output_Results/Partitioned_Models/predictions_table.csv
* Evaluation_Script_4.ipynb
o No output file saved.?
o Results are contained within the notebook for evaluating different cuts.
Operationalize Modeling
This code is ready to be productionized. Input data files need to be in the exact same format as were shared with us by the project team. Change the following parameters to run the modeling pipeline for latest years or can leave as is i.e 2019 in Scoring_rand that was used for development:
* Data_Preparation_Script_1.ipynb
o Training_rand = "2018"
o Scoring_rand?= "2019" 
* OR "2017" OR for latest year if the data is available
* data_file = "rand_2018.csv" 
o Year mentioned in this parameter for input file depends on the year mentioned in Data_Preparation_Script_1.ipynb. Currently set for 2019.
* Scoring_Script_3.ipynb
o data_file = "rand_2019.csv" OR "rand_2017.csv"
o Year mentioned in this parameter for input file depends on the year mentioned in Data_Preparation_Script_1.ipynb. Currently set for 2019.

We have tested with scoring data of 2019 and 2017.

Currently the existing code is ready to run without changing any parameter. Schedule or Run the notebooks in the following sequence:
1. Data_Preparation_Script_1.ipynb
2. Modeling_Script_2.ipynb
3. Scoring_Script_3.ipynb
4. Evaluation_Script_4.ipynb
