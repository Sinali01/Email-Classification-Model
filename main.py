import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# reading the file
df1 = pd.read_csv('spambase/spambase.data')
print(df1)

with open('spambase/spambase.names') as spam:
    text = spam.read()
# finding the pattern
labels = re.findall(r'\n(\w*_?\W?):', text)
SpamData = pd.read_csv('spambase/spambase.data', header=None, names=labels +['spam'])
print(SpamData)

# counting the number of boolean values and creating a boolean dataframe and adding up the 'true' values
SpamData_missing = SpamData.isna().sum()
print(SpamData_missing)

df2 = pd.DataFrame(SpamData)
print(df2)

# dropping unnecessary columns
df2 = df2.drop(labels=['word_freq_3d', 'char_freq_['],axis=1)
print(df2)

# checking duplicates
print(df2.duplicated())

# dropping the duplicates which says 'True'
df2.drop_duplicates(inplace=True)
print(df2)

# Storing the all values of 56th column in y array
y = df2.iloc[:, 55].values

# providing the summary of descriptive statistics count, mean, standard deviation, minimum and maximum values,
# as well as the 25th, 50th, and 75th percentiles of the data
df2_description = df2.describe()
print(df2_description)
