#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 20:12:22 2020

@author: hyunwoo
"""

import pandas as pd
import numpy as np
import seaborn as sns
from patsy import dmatrices
import statsmodels.api as sm;
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import t, f, chi2, skew, kurtosis
from statsmodels.stats.outliers_influence import variance_inflation_factor

data=pd.read_csv('https://drive.google.com/uc?export=download&id=1ssBNxmds4zmmJbAHzJUB0_UyyfyMtoHT')

from statsmodels.formula.api import ols

model1 = ols('y ~ Po1 + Po2 + GDP + Prob + Pop ',data)
res1 = model1.fit()
res1.summary()

model2 = ols('y ~  GDP + Prob + Pop',data)

res2 = model2.fit()
res2.summary()

model2.exog_names
VIF = []
for i in range(1,4):
    VIF.append(variance_inflation_factor(model2.exog,i))