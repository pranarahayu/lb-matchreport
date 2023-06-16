from sklearn import preprocessing
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st

import os
import pandas as pd
import glob
from datetime import date
import numpy as np

df = pd.read_excel(st.secrets["public_gsheets_url"])

#converting categorical to numerical columns
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(df["body_part"])
df['body_part_num'] = label_encoder.transform(df["body_part"])

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(df["situation"])
df['situation_num'] = label_encoder.transform(df["situation"])

##Variables
model_variables = ['angledeg','distance','body_part_num','situation_num']
model=''
for v in model_variables[:-1]:
    model = model  + v + ' + '
model = model + model_variables[-1]

##Modeling 1
test_model = smf.glm(formula="goal ~ " + model, data=df, 
                           family=sm.families.Binomial()).fit()
print(test_model.summary())        
b=test_model.params

#Creating xG function to calculate the xG
def calculate_xG(sh):    
   bsum=b[0]
   for i,v in enumerate(model_variables):
       bsum=bsum+b[i+1]*sh[v]
   xG = 1/(1+np.exp(bsum)) 
   return xG   

xG=df.apply(calculate_xG, axis=1) 
df = df.assign(xG=xG)

#Other than penalties, all xG seems weird. This function fix that
def xgfix(row):
  if row['situation'] == 'Penalty':
    return row['xG']
  else:
    return 1-row['xG']
df['xG'] = df.apply(lambda row: xgfix(row), axis=1)
