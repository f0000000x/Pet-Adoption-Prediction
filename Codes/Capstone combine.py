# Libraries ========================================================================================================== #
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go

import cv2
import os
import time
import gc
import glob
import json
import pprint
import joblib
import warnings
import random

import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import tensorflow as tf

from collections import Counter
from functools import partial
from math import sqrt
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import SparsePCA, TruncatedSVD, LatentDirichletAllocation, NMF

from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm
from contextlib import contextmanager
from pandas.io.json import json_normalize

# Comment this if the data visualisations doesn't work on your side
%matplotlib inline

df_EDA = pd.read_csv("train.csv")

pd.set_option('display.max_columns', None)

df_EDA.head(3)

# duplicate values
print(df_EDA.duplicated())
# missing values
print(df_EDA.isnull().any())
# number of missing values
for COL in df_EDA.columns:
    print(COL + ':', len(df_EDA) - df_EDA[COL].count())

# drop effect-free columns
df_EDA = df_EDA.drop(['Name', 'RescuerID', 'Description', 'PetID'], axis=1)

# drop missing values
df_EDA = df_EDA.dropna()
# drop duplicated values
df_EDA = df_EDA.drop_duplicates()
print("Orginal number of observations: 14993")
print("Number of cleaned observations:", len(df_EDA))
print("Omitted observations          :", 14993 - len(df_EDA))

# Pet Type
df_EDA['Type'] = df_EDA['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')
# Gender
df_EDA['Gender']=df_EDA['Gender'].apply(lambda x: 'Male'if x == 1 else ('Female' if x == 2 else 'Not Sure'))
# Matrity Size
df_EDA['MaturitySize']=df_EDA['MaturitySize'].apply(lambda x: 'Not Specified'if x == 0 else ('Small' if x == 1 else ('Medium' if x == 2 else ('Large' if x == 3 else 'Extra Large' ))))
# Fur Length
df_EDA['FurLength']=df_EDA['FurLength'].apply(lambda x: 'Not Specified'if x == 0 else ("Short" if x == 1 else ('Medium' if x == 2 else "Long")))
# Vaccinated
df_EDA['Vaccinated']=df_EDA['Vaccinated'].apply(lambda x: "Yes"if x == 1 else ("No" if x == 2 else "Not Sure"))
# Dewormed
df_EDA['Dewormed']=df_EDA['Dewormed'].apply(lambda x: "Yes"if x == 1 else ("No" if x == 2 else "Not Sure"))
# Sterilized
df_EDA['Sterilized']=df_EDA['Sterilized'].apply(lambda x: "Yes"if x == 1 else ("No" if x == 2 else "Not Sure"))
# Health
df_EDA['Health']=df_EDA['Health'].apply(lambda x: "Healthy" if x == 1 else ("Minor Injury" if x == 2 else ("Serious Injury" if x == 3 else "Not Sure")))
#Adoption Speed
df_EDA['AdoptionSpeed']=df_EDA['AdoptionSpeed'].apply(lambda x: "Same Day"if x == 0 else ("1-7 Days" if x == 1 else ("8-30 Days" if x == 2 else ("31-90 Days" if x == 3 else "No Adoption" ))))

#The target variable: Adoption Speed
sns.set_palette("RdBu_r")
f, ax = plt.subplots(figsize=(7, 4))
plt.title('Adoption Speed classes counts')
sns.countplot(y="AdoptionSpeed", data=df_EDA,order=["Same Day","1-7 Days", "8-30 Days", "31-90 Days","No Adoption"])

counts_target = df_EDA['AdoptionSpeed'].value_counts().sort_index()
target_percentage = df_EDA['AdoptionSpeed'].value_counts(normalize=True).sort_index().mul(100).round(1).astype(str) + "%"
target_discription = ["adopted on the same day","adopted between 1 and 7 days","adopted between 8 and 30 days","between 31 and 90 days","No adoption after 100 days"]
pd.DataFrame({'Counts':counts_target,'Percentage':target_percentage}).reindex(['Same Day','1-7 Days','8-30 Days','31-90 Days','No Adoption'])

#Dogs versus Cats
fig, ax = plt.subplots(figsize=(25, 7))

plt.subplot(1, 3, 1)
plt.title('Dogs versus Cats')
type_plot = sns.countplot(x='Type', data=df_EDA)
ax = type_plot.axes
for p in ax.patches:
    ax.annotate(f"{p.get_height() * 100 / df_EDA.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, rotation=0, xytext=(0, 10),
                textcoords='offset points')
plt.legend()

plt.subplot(1, 3, 2)
plt.title('Adoption for Each Type: Cat')
Only_Cat = df_EDA.loc[df_EDA['Type'] == 'Cat']
type_cat_plot = sns.countplot(x="Type", hue="AdoptionSpeed",
                              hue_order=["Same Day", "1-7 Days", "8-30 Days", "31-90 Days", "No Adoption"],
                              data=Only_Cat)
ax = type_cat_plot.axes
for p in ax.patches:
    ax.annotate(f"{p.get_height() * 100 / Only_Cat.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, rotation=0, xytext=(0, 10),
                textcoords='offset points')

plt.legend()

plt.subplot(1, 3, 3)
plt.title('Adoption for Each Type: Dog')
Only_Dog = df_EDA.loc[df_EDA['Type'] == 'Dog']
type_Dog_plot = sns.countplot(x="Type", hue="AdoptionSpeed",
                              hue_order=["Same Day", "1-7 Days", "8-30 Days", "31-90 Days", "No Adoption"],
                              data=Only_Dog)
ax = type_Dog_plot.axes
for p in ax.patches:
    ax.annotate(f"{p.get_height() * 100 / Only_Cat.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, rotation=0, xytext=(0, 10),
                textcoords='offset points')

sns.despine(left=True, bottom=True)
plt.legend()

#Age versus Adoption Rate
# Add histogram data
x1 = df_EDA.loc[df_EDA["AdoptionSpeed"] == "Same Day"]['Age']
x2 = df_EDA.loc[df_EDA["AdoptionSpeed"] == "1-7 Days"]['Age']
x3 = df_EDA.loc[df_EDA["AdoptionSpeed"] == "8-30 Days"]['Age']
x4 = df_EDA.loc[df_EDA["AdoptionSpeed"] == "31-90 Days"]['Age']
x5 = df_EDA.loc[df_EDA["AdoptionSpeed"] == "No Adoption"]['Age']

# Group data together
hist_data = [x1, x2, x3, x4, x5]

group_labels = ["Same Day", "1-7 Days", "8-30 Days", "31-90 Days", "No Adoption"]

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, show_hist=False)
fig.update_layout(height=700,width=800)
fig.update_layout(title_text="AdoptionSpeed trends by Age",
                  xaxis=dict(
                      title_text="Age(in Months)"))
#kernel density estimation

# Bin_Ranges = {'Bins Range':['0-6', '6-12','12-36','36-60','60-96','96-255']}
df_EDA['AgeBins'] = 6
df_EDA.loc[df_EDA['Age']< 6, 'AgeBins'] = 1
df_EDA.loc[(df_EDA['Age'] >= 6)&(df_EDA['Age'] < 12), 'AgeBins'] = 2
df_EDA.loc[(df_EDA['Age'] >= 12)&(df_EDA['Age'] < 36), 'AgeBins'] = 3
df_EDA.loc[(df_EDA['Age'] >= 36)&(df_EDA['Age'] < 60), 'AgeBins'] = 4
df_EDA.loc[(df_EDA['Age'] >= 60)&(df_EDA['Age'] < 96), 'AgeBins'] = 5

#print("Creating Six Bins:")
Bins = pd.DataFrame(data=df_EDA["AgeBins"].value_counts().rename_axis('Bin Label').reset_index(name='counts'))
Bins.insert(loc=1, column='Bin Range', value=['0 - 6', '6 - 12','12 - 36','36 - 60','60 - 96','96 - 255'])


b1 = df_EDA[(df_EDA["AdoptionSpeed"] == "Same Day") & (df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Dog')].shape[0]
b2 = df_EDA[(df_EDA["AdoptionSpeed"] == "1-7 Days") & (df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Dog')].shape[0]
b3 = df_EDA[(df_EDA["AdoptionSpeed"] == "8-30 Days") & (df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Dog')].shape[0]
b4 = df_EDA[(df_EDA["AdoptionSpeed"] == "31-90 Days") & (df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Dog')].shape[0]
b5 = df_EDA[(df_EDA["AdoptionSpeed"] == "No Adoption") & (df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Dog')].shape[0]

b6 = df_EDA[(df_EDA["AdoptionSpeed"] == "Same Day") & (df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Dog')].shape[0]
b7 = df_EDA[(df_EDA["AdoptionSpeed"] == "1-7 Days") & (df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Dog')].shape[0]
b8 = df_EDA[(df_EDA["AdoptionSpeed"] == "8-30 Days") & (df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Dog')].shape[0]
b9 = df_EDA[(df_EDA["AdoptionSpeed"] == "31-90 Days") & (df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Dog')].shape[0]
b10 = df_EDA[(df_EDA["AdoptionSpeed"] == "No Adoption") & (df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Dog')].shape[0]

b11 = df_EDA[(df_EDA["AdoptionSpeed"] == "Same Day") & (df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Dog')].shape[0]
b12 = df_EDA[(df_EDA["AdoptionSpeed"] == "1-7 Days") & (df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Dog')].shape[0]
b13 = df_EDA[(df_EDA["AdoptionSpeed"] == "8-30 Days") & (df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Dog')].shape[0]
b14 = df_EDA[(df_EDA["AdoptionSpeed"] == "31-90 Days") & (df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Dog')].shape[0]
b15 = df_EDA[(df_EDA["AdoptionSpeed"] == "No Adoption") & (df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Dog')].shape[0]

b16 = df_EDA[(df_EDA["AdoptionSpeed"] == "Same Day") & (df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Dog')].shape[0]
b17 = df_EDA[(df_EDA["AdoptionSpeed"] == "1-7 Days") & (df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Dog')].shape[0]
b18 = df_EDA[(df_EDA["AdoptionSpeed"] == "8-30 Days") & (df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Dog')].shape[0]
b19 = df_EDA[(df_EDA["AdoptionSpeed"] == "31-90 Days") & (df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Dog')].shape[0]
b20 = df_EDA[(df_EDA["AdoptionSpeed"] == "No Adoption") & (df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Dog')].shape[0]

b21 = df_EDA[(df_EDA["AdoptionSpeed"] == "Same Day") & (df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Dog')].shape[0]
b22 = df_EDA[(df_EDA["AdoptionSpeed"] == "1-7 Days") & (df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Dog')].shape[0]
b23 = df_EDA[(df_EDA["AdoptionSpeed"] == "8-30 Days") & (df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Dog')].shape[0]
b24 = df_EDA[(df_EDA["AdoptionSpeed"] == "31-90 Days") & (df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Dog')].shape[0]
b25 = df_EDA[(df_EDA["AdoptionSpeed"] == "No Adoption") & (df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Dog')].shape[0]

b26 = df_EDA[(df_EDA["AdoptionSpeed"] == "Same Day") & (df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Dog')].shape[0]
b27 = df_EDA[(df_EDA["AdoptionSpeed"] == "1-7 Days") & (df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Dog')].shape[0]
b28 = df_EDA[(df_EDA["AdoptionSpeed"] == "8-30 Days") & (df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Dog')].shape[0]
b29 = df_EDA[(df_EDA["AdoptionSpeed"] == "31-90 Days") & (df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Dog')].shape[0]
b30 = df_EDA[(df_EDA["AdoptionSpeed"] == "No Adoption") & (df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Dog')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Dog')].shape[0]

x=['0 - 6', '6 - 12','12 - 36','36 - 60','60 - 96','96 - 255']
fig = go.Figure(go.Bar(x=x, y=[b1,b6,b11,b16,b21,b26], name="Same Day"))
fig.add_trace(go.Bar(x=x, y=[b2,b7,b12,b17,b22,b27], name="1-7 Days"))
fig.add_trace(go.Bar(x=x, y=[b3,b8,b13,b18,b23,b28], name="8-30 Days"))
fig.add_trace(go.Bar(x=x, y=[b4,b9,b14,b19,b24,b29], name="31-90 Days"))
fig.add_trace(go.Bar(x=x, y=[b5,b10,b15,b20,b25,b30], name="No Adoption"))

fig.update_layout(barmode='stack', title="Age Bins for Dog",xaxis_title="Age Bins(in months)")
fig.show()

c1 = df_EDA[(df_EDA["AdoptionSpeed"] == "Same Day") & (df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Cat')].shape[0]
c2 = df_EDA[(df_EDA["AdoptionSpeed"] == "1-7 Days") & (df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Cat')].shape[0]
c3 = df_EDA[(df_EDA["AdoptionSpeed"] == "8-30 Days") & (df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Cat')].shape[0]
c4 = df_EDA[(df_EDA["AdoptionSpeed"] == "31-90 Days") & (df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Cat')].shape[0]
c5 = df_EDA[(df_EDA["AdoptionSpeed"] == "No Adoption") & (df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 1) & (df_EDA['Type'] == 'Cat')].shape[0]

c6 = df_EDA[(df_EDA["AdoptionSpeed"] == "Same Day") & (df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Cat')].shape[0]
c7 = df_EDA[(df_EDA["AdoptionSpeed"] == "1-7 Days") & (df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Cat')].shape[0]
c8 = df_EDA[(df_EDA["AdoptionSpeed"] == "8-30 Days") & (df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Cat')].shape[0]
c9 = df_EDA[(df_EDA["AdoptionSpeed"] == "31-90 Days") & (df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Cat')].shape[0]
c10 = df_EDA[(df_EDA["AdoptionSpeed"] == "No Adoption") & (df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 2) & (df_EDA['Type'] == 'Cat')].shape[0]

c11 = df_EDA[(df_EDA["AdoptionSpeed"] == "Same Day") & (df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Cat')].shape[0]
c12 = df_EDA[(df_EDA["AdoptionSpeed"] == "1-7 Days") & (df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Cat')].shape[0]
c13 = df_EDA[(df_EDA["AdoptionSpeed"] == "8-30 Days") & (df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Cat')].shape[0]
c14 = df_EDA[(df_EDA["AdoptionSpeed"] == "31-90 Days") & (df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Cat')].shape[0]
c15 = df_EDA[(df_EDA["AdoptionSpeed"] == "No Adoption") & (df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 3) & (df_EDA['Type'] == 'Cat')].shape[0]

c16 = df_EDA[(df_EDA["AdoptionSpeed"] == "Same Day") & (df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Cat')].shape[0]
c17 = df_EDA[(df_EDA["AdoptionSpeed"] == "1-7 Days") & (df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Cat')].shape[0]
c18 = df_EDA[(df_EDA["AdoptionSpeed"] == "8-30 Days") & (df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Cat')].shape[0]
c19 = df_EDA[(df_EDA["AdoptionSpeed"] == "31-90 Days") & (df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Cat')].shape[0]
c20 = df_EDA[(df_EDA["AdoptionSpeed"] == "No Adoption") & (df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 4) & (df_EDA['Type'] == 'Cat')].shape[0]

c21 = df_EDA[(df_EDA["AdoptionSpeed"] == "Same Day") & (df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Cat')].shape[0]
c22 = df_EDA[(df_EDA["AdoptionSpeed"] == "1-7 Days") & (df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Cat')].shape[0]
c23 = df_EDA[(df_EDA["AdoptionSpeed"] == "8-30 Days") & (df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Cat')].shape[0]
c24 = df_EDA[(df_EDA["AdoptionSpeed"] == "31-90 Days") & (df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Cat')].shape[0]
c25 = df_EDA[(df_EDA["AdoptionSpeed"] == "No Adoption") & (df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 5) & (df_EDA['Type'] == 'Cat')].shape[0]

c26 = df_EDA[(df_EDA["AdoptionSpeed"] == "Same Day") & (df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Cat')].shape[0]
c27 = df_EDA[(df_EDA["AdoptionSpeed"] == "1-7 Days") & (df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Cat')].shape[0]
c28 = df_EDA[(df_EDA["AdoptionSpeed"] == "8-30 Days") & (df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Cat')].shape[0]
c29 = df_EDA[(df_EDA["AdoptionSpeed"] == "31-90 Days") & (df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Cat')].shape[0]
c30 = df_EDA[(df_EDA["AdoptionSpeed"] == "No Adoption") & (df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Cat')].shape[0]/df_EDA[(df_EDA['AgeBins'] == 6) & (df_EDA['Type'] == 'Cat')].shape[0]

x=['0 - 6', '6 - 12','12 - 36','36 - 60','60 - 96','96 - 255']
fig = go.Figure(go.Bar(x=x, y=[c1,c6,c11,c16,c21,c26], name="Same Day"))
fig.add_trace(go.Bar(x=x, y=[c2,c7,c12,c17,c22,c27], name="1-7 Days"))
fig.add_trace(go.Bar(x=x, y=[c3,c8,c13,c18,c23,c28], name="8-30 Days"))
fig.add_trace(go.Bar(x=x, y=[c4,c9,c14,c19,c24,c29], name="31-90 Days"))
fig.add_trace(go.Bar(x=x, y=[c5,c10,c15,c20,c25,c30], name="No Adoption"))

fig.update_layout(barmode='stack', title="Age Bins for Cat",xaxis_title="Age Bins(in months)")
fig.show()

#Breeds
df_EDA['Pure_breed'] = "Mix"
df_EDA.loc[df_EDA['Breed2'] == 0, 'Pure_breed'] = "Pure"

a1 = df_EDA[(df_EDA["AdoptionSpeed"] == "Same Day") & (df_EDA['Pure_breed'] == "Mix")].shape[0]/ df_EDA[df_EDA["AdoptionSpeed"] == "Same Day"].shape[0]
a2 = df_EDA[(df_EDA["AdoptionSpeed"] == "Same Day") & (df_EDA['Pure_breed'] == "Pure")].shape[0]/ df_EDA[df_EDA["AdoptionSpeed"] == "Same Day"].shape[0]
a3 = df_EDA[(df_EDA["AdoptionSpeed"] == "1-7 Days") & (df_EDA['Pure_breed'] == "Mix")].shape[0]/ df_EDA[df_EDA["AdoptionSpeed"] == "1-7 Days"].shape[0]
a4 = df_EDA[(df_EDA["AdoptionSpeed"] == "1-7 Days") & (df_EDA['Pure_breed'] == "Pure")].shape[0]/ df_EDA[df_EDA["AdoptionSpeed"] == "1-7 Days"].shape[0]
a5 = df_EDA[(df_EDA["AdoptionSpeed"] == "8-30 Days") & (df_EDA['Pure_breed'] == "Mix")].shape[0]/ df_EDA[df_EDA["AdoptionSpeed"] == "8-30 Days"].shape[0]
a6 = df_EDA[(df_EDA["AdoptionSpeed"] == "8-30 Days") & (df_EDA['Pure_breed'] == "Pure")].shape[0]/ df_EDA[df_EDA["AdoptionSpeed"] == "8-30 Days"].shape[0]
a7 = df_EDA[(df_EDA["AdoptionSpeed"] == "31-90 Days") & (df_EDA['Pure_breed'] == "Mix")].shape[0]/ df_EDA[df_EDA["AdoptionSpeed"] == "31-90 Days"].shape[0]
a8 = df_EDA[(df_EDA["AdoptionSpeed"] == "31-90 Days") & (df_EDA['Pure_breed'] == "Pure")].shape[0]/ df_EDA[df_EDA["AdoptionSpeed"] == "31-90 Days"].shape[0]
a9 = df_EDA[(df_EDA["AdoptionSpeed"] == "No Adoption") & (df_EDA['Pure_breed'] == "Mix")].shape[0]/ df_EDA[df_EDA["AdoptionSpeed"] == "No Adoption"].shape[0]
a10 = df_EDA[(df_EDA["AdoptionSpeed"] =="No Adoption") & (df_EDA['Pure_breed'] == "Pure")].shape[0]/ df_EDA[df_EDA["AdoptionSpeed"] == "No Adoption"].shape[0]
y=["Same Day", "1-7 Days", "8-30 Days", "31-90 Days","No Adoption"]


fig, ax = plt.subplots(figsize = (15, 5))
plt.subplot(1, 2, 1)
plt.title('Pure_breed: Dogs vs Cats');
g=sns.countplot(x='Pure_breed', data=df_EDA);
ax=g.axes
for p in ax.patches:
     ax.annotate(f"{p.get_height() * 100 / df_EDA.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, rotation=0, xytext=(0, 10),
         textcoords='offset points')
plt.legend()

plt.subplot(1, 2, 2)
# Plot the total in each speed
sns.set_color_codes("pastel")
sns.barplot(x=[a1+a2, a3+a4, a5+a6, a7+a8,a9+a10], y=y,
            label="Mix", color="b")

# Plot the
sns.set_color_codes("muted")
sns.barplot(x=[a2,a4,a6,a8,a10], y=y,
            label="Pure", color="b")

# Add a legend and informative axis label
plt.title('Percentage of Dogs vs Cats in each AdoptionSpeed')
sns.despine(left=True, bottom=True)
plt.legend()

#Gender
fig, ax = plt.subplots(figsize=(30, 8))

plt.subplot(1, 4, 1)
Gender_plot = sns.countplot(x='Gender', data=df_EDA)
ax = Gender_plot.axes
for p in ax.patches:
    ax.annotate(f"{p.get_height() * 100 / df_EDA.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, rotation=0, xytext=(0, 10),
                textcoords='offset points')
plt.legend()

plt.subplot(1, 4, 2)
plt.title('Adoption Speed: Male')
Only_Male = df_EDA.loc[df_EDA['Gender'] == 'Male']
Only_Male_plot = sns.countplot(x='Gender', hue="AdoptionSpeed",
                               hue_order=["Same Day", "1-7 Days", "8-30 Days", "31-90 Days", "No Adoption"],
                               data=Only_Male)
ax = Only_Male_plot.axes
for p in ax.patches:
    ax.annotate(f"{p.get_height() * 100 / Only_Male.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, rotation=0, xytext=(0, 10),
                textcoords='offset points')

plt.legend()

plt.subplot(1, 4, 3)
plt.title('Adoption Speed: Female')
Only_Female = df_EDA.loc[df_EDA['Gender'] == 'Female']
Only_Female_plot = sns.countplot(x='Gender', hue="AdoptionSpeed",
                                 hue_order=["Same Day", "1-7 Days", "8-30 Days", "31-90 Days", "No Adoption"],
                                 data=Only_Female)
ax = Only_Female_plot.axes
for p in ax.patches:
    ax.annotate(f"{p.get_height() * 100 / Only_Female.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, rotation=0, xytext=(0, 10),
                textcoords='offset points')
plt.legend()

plt.subplot(1, 4, 4)
plt.title('Adoption Speed: Not Sure')
Only_Mixed = df_EDA.loc[df_EDA['Gender'] == 'Not Sure']
Only_Mixed_plot = sns.countplot(x='Gender', hue="AdoptionSpeed",
                                hue_order=["Same Day", "1-7 Days", "8-30 Days", "31-90 Days", "No Adoption"],
                                data=Only_Mixed)
ax = Only_Mixed_plot.axes
for p in ax.patches:
    ax.annotate(f"{p.get_height() * 100 / Only_Mixed.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, rotation=0, xytext=(0, 10),
                textcoords='offset points')

sns.despine(left=True, bottom=True)
plt.legend()

#Adoption Fee
df_EDA['Free_or_NeedFee'] = df_EDA['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
fig, ax = plt.subplots(figsize=(25, 7))

plt.subplot(1, 3, 1)
Free_or_NeedFee_plot = sns.countplot(x='Free_or_NeedFee', data=df_EDA)
ax = Free_or_NeedFee_plot.axes
for p in ax.patches:
    ax.annotate(f"{p.get_height() * 100 / df_EDA.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, rotation=0, xytext=(0, 10),
                textcoords='offset points')
plt.legend()

plt.subplot(1, 3, 2)
plt.title('Adoption Speed: Free')
Free_set = df_EDA.loc[df_EDA['Free_or_NeedFee'] == 'Free']
Free_set_plot = sns.countplot(x='Free_or_NeedFee', hue="AdoptionSpeed",
                              hue_order=["Same Day", "1-7 Days", "8-30 Days", "31-90 Days", "No Adoption"],
                              data=Free_set)
ax = Free_set_plot.axes
for p in ax.patches:
    ax.annotate(f"{p.get_height() * 100 / Free_set.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, rotation=0, xytext=(0, 10),
                textcoords='offset points')

plt.legend()

plt.subplot(1, 3, 3)
plt.title('Adoption Speed: Not Free')
Fee_set = df_EDA.loc[df_EDA['Free_or_NeedFee'] == 'Not Free']
Fee_set_plot = sns.countplot(x='Free_or_NeedFee', hue="AdoptionSpeed",
                             hue_order=["Same Day", "1-7 Days", "8-30 Days", "31-90 Days", "No Adoption"], data=Fee_set)
ax = Fee_set_plot.axes
for p in ax.patches:
    ax.annotate(f"{p.get_height() * 100 / Fee_set.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, rotation=0, xytext=(0, 10),
                textcoords='offset points')

sns.despine(left=True, bottom=True)
plt.legend()

plt.figure(figsize=(16, 6));
plt.subplot(1, 2, 1)
plt.hist(df_EDA.loc[df_EDA['Fee'] < 500, 'Fee'])
plt.title('Distribution of fees less than $500')

plt.subplot(1, 2, 2)
sns.violinplot(x="AdoptionSpeed", y="Fee", hue="Type", data=df_EDA,order=["Same Day","1-7 Days", "8-30 Days", "31-90 Days","No Adoption"])
plt.title('AdoptionSpeed by Type and Fee')

plt.figure(figsize=(8, 5));
sns.scatterplot(x="Fee", y="Quantity", hue="Type",data=df_EDA);
plt.title('Quantity of pets vs Fee');


#Preprocess
#Type
all_data['Type'] = all_data['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')
#Name
train['Name'] = train['Name'].fillna('Unnamed')
test['Name'] = test['Name'].fillna('Unnamed')
all_data['Name'] = all_data['Name'].fillna('Unnamed')
train['No_name'] = 0
train.loc[train['Name'] == 'Unnamed', 'No_name'] = 1
test['No_name'] = 0
test.loc[test['Name'] == 'Unnamed', 'No_name'] = 1
all_data['No_name'] = 0
all_data.loc[all_data['Name'] == 'Unnamed', 'No_name'] = 1
#Age

#Breeds
breeds_dict = {k: v for k, v in zip(breeds['BreedID'], breeds['BreedName'])}

train['Pure_breed'] = 0
train.loc[train['Breed2'] == 0, 'Pure_breed'] = 1
test['Pure_breed'] = 0
test.loc[test['Breed2'] == 0, 'Pure_breed'] = 1
all_data['Pure_breed'] = 0
all_data.loc[all_data['Breed2'] == 0, 'Pure_breed'] = 1

train['Breed1_name'] = train['Breed1'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'Unknown')
train['Breed2_name'] = train['Breed2'].apply(lambda x: '_'.join(breeds_dict[x]) if x in breeds_dict else '-')

test['Breed1_name'] = test['Breed1'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'Unknown')
test['Breed2_name'] = test['Breed2'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else '-')

all_data['Breed1_name'] = all_data['Breed1'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'Unknown')
all_data['Breed2_name'] = all_data['Breed2'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else '-')

#Gender

#Color
colors_dict = {k: v for k, v in zip(colors['ColorID'], colors['ColorName'])}
train['Color1_name'] = train['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
train['Color2_name'] = train['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
train['Color3_name'] = train['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

test['Color1_name'] = test['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
test['Color2_name'] = test['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
test['Color3_name'] = test['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

all_data['Color1_name'] = all_data['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
all_data['Color2_name'] = all_data['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
all_data['Color3_name'] = all_data['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

train['full_color'] = (train['Color1_name'] + '__' + train['Color2_name'] + '__' + train['Color3_name']).str.replace('__', '')
test['full_color'] = (test['Color1_name'] + '__' + test['Color2_name'] + '__' + test['Color3_name']).str.replace('__', '')
all_data['full_color'] = (all_data['Color1_name'] + '__' + all_data['Color2_name'] + '__' + all_data['Color3_name']).str.replace('__', '')

#MatiritySize

#FurLength

#Health
train['health'] = train['Vaccinated'].astype(str) + '_' + train['Dewormed'].astype(str) + '_' + train['Sterilized'].astype(str) + '_' + train['Health'].astype(str)
test['health'] = test['Vaccinated'].astype(str) + '_' + test['Dewormed'].astype(str) + '_' + test['Sterilized'].astype(str) + '_' + test['Health'].astype(str)

#Quantity
train['Quantity_short'] = train['Quantity'].apply(lambda x: x if x <= 5 else 6)
test['Quantity_short'] = test['Quantity'].apply(lambda x: x if x <= 5 else 6)
all_data['Quantity_short'] = all_data['Quantity'].apply(lambda x: x if x <= 5 else 6)

#Fee
train['Free'] = train['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
test['Free'] = test['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
all_data['Free'] = all_data['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')

#State
states_dict = {k: v for k, v in zip(states['StateID'], states['StateName'])}
train['State_name'] = train['State'].apply(lambda x: '_'.join(states_dict[x].split()) if x in states_dict else 'Unknown')
test['State_name'] = test['State'].apply(lambda x: '_'.join(states_dict[x].split()) if x in states_dict else 'Unknown')
all_data['State_name'] = all_data['State'].apply(lambda x: '_'.join(states_dict[x].split()) if x in states_dict else 'Unknown')

#Rescuer
#VideoAmt
#PhotoAmt

#Description
train['Description'] = train['Description'].fillna('')
test['Description'] = test['Description'].fillna('')
all_data['Description'] = all_data['Description'].fillna('')

train['desc_length'] = train['Description'].apply(lambda x: len(x))
train['desc_words'] = train['Description'].apply(lambda x: len(x.split()))

test['desc_length'] = test['Description'].apply(lambda x: len(x))
test['desc_words'] = test['Description'].apply(lambda x: len(x.split()))

all_data['desc_length'] = all_data['Description'].apply(lambda x: len(x))
all_data['desc_words'] = all_data['Description'].apply(lambda x: len(x.split()))

train['averate_word_length'] = train['desc_length'] / train['desc_words']
test['averate_word_length'] = test['desc_length'] / test['desc_words']
all_data['averate_word_length'] = all_data['desc_length'] / all_data['desc_words']