import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import sklearn

train_path = "/Users/theodoreplotkin/desktop/postmalone/GA_Data_Science/DAT-06-24/class material/Unit 3/data/iowa_housing/train.csv"
test_path = "/Users/theodoreplotkin/desktop/postmalone/GA_Data_Science/DAT-06-24/class material/Unit 3/data/iowa_housing/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

numeric_features = train.dtypes[train.dtypes != "object"].index

train_cts = train[numeric_features]

print(train_cts)
print(train_cts.columns)



#1) FEATURE PREPERATION
#goal: want an algorithm that creates a design matrix X
##     from train dataset, with the following properties:

#overall features TODO::
#-drop massively NA features (ex/ PoolQC, Alley)
#-impute missing values
#-standardize

    #cts features TODO:: 
    #-fix the skew of certain features
    #-create a totalsqft/totalporch/totalbsmt/total bathroom feature
    #-create "is new" feature if yearsold == yearbuilt
    #-create "house age" feature as year sold - year remolded
    #remove major outliers 

    #ordered categorical features TODO::
    #-map features to ordinals
    #-check for less than 25 per category 

    #unordered categorical features TODO::
    #-apply "one-hot" encoding
    #-check for less than 25 per category 

#order
#1st -- DATA CLEANING
##apply the log transform to the CTS variables (those with skew > 0.75)

##do appropriate ordinal transforms to ordered discrete variables
##do one-hot encoding to unordered discrete variables
##remove outliers
##notice NA's -- impute missing


#2nd -- FEATURE ENGINEERING
#examine the correlations between obviously related features and interact appropriately
##this will reduce the number of columns a bit 

#3rd -- FEATURE SELECTION
#try lasso/elastic-net


#4th -- MODEL TESTING/gridsearchCV
#XGBoost
#KernelRidge
#Ridge
#Lasso
#ElasticNet
#RandomForest

#5th -- Conclusions

