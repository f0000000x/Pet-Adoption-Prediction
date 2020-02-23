# Libraries ========================================================================================================== #
import pandas as pd

# Data reading ======================================================================================================= #
path = "/Users/Fox/Desktop/Data Science/Capstone"
df = pd.read_csv(path + "/Data/train/train.csv")
# print(df)

# Check Data Quality ================================================================================================= #
# duplicate values
print(df.duplicated())
# missing values
print(df.isnull().any())
# number of missing values
for COL in df.columns:
    print(COL + ':', len(df) - df[COL].count())

# drop effect-free columns
df = df.drop(['Name', 'RescuerID', 'Description', 'PetID'], axis=1)

# drop missing values
df = df.dropna()
# drop duplicated values
df = df.drop_duplicates()
print("Orginal number of observations: 14993")
print("Number of cleaned observations:", len(df))
print("Omitted observations          :", 14993 - len(df))

# data skew check
print(pd.value_counts(df['AdoptionSpeed']))

# outlier check
# calculate relevant statistical indicators
# get descriptive statistics
statDF = df.describe()
# get the maximum and minimum values of each field
Max_min = statDF.loc[['max', 'min']]
# update statDF
statDF = df.describe()
# calculate the mean + triple standard deviations
statDF.loc['mean+3std'] = statDF.loc['mean'] + 3 * statDF.loc['std']
# calculate the mean - triple standard deviations
statDF.loc['mean-3std'] = statDF.loc['mean'] - 3 * statDF.loc['std']
# calculate the upper quartile + 1.5 times quartile spacing
statDF.loc['75%+1.5dist'] = statDF.loc['75%'] + 1.5 * (statDF.loc['75%'] - statDF.loc['25%'])
# calculate the lower quartile + 1.5 times quartile spacing
statDF.loc['25%-1.5dist'] = statDF.loc['25%'] - 1.5 * (statDF.loc['75%'] - statDF.loc['25%'])
# let's see if it's greater than the mean + 3 standard deviations
dataset3 = df - statDF.loc['mean+3std']
print(dataset3[dataset3 > 0])
# let's see if it's fewer than the mean - 3 standard deviations
dataset4 = df - statDF.loc['mean-3std']
print(dataset4[dataset4 < 0])
# if it's greater than the upper quartile + 1.5 times the quartile spacing
dataset5 = df - statDF.loc['75%+1.5dist']
print(dataset5[dataset5 > 0])
# if it's fewer than the upper quartile - 1.5 times the quartile spacing
dataset6 = df - statDF.loc['25%-1.5dist']
print(dataset6[dataset6 < 0])

# save the cleaned data
df.to_csv(path + "/Codes/DataPre.csv", index=None)