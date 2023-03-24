import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# reading the file
df1 = pd.read_csv('spambase/spambase.data')
with open('spambase/spambase.data') as spam:
    text = spam.read()
print(df1)

# finding the pattern
labels = re.findall(r'\n(\w*_?\W?):', text)
SpamData = pd.read_csv('spambase/spambase.data', header=None, names=labels +['spam'])
print(SpamData)

# creating a boolean dataframe and adding up the 'true' values
SpamData.isna().sum()

df2 = SpamData
print(df2)

# removing 2 columns
df2 = df2.drop(labels=['word_freq_3d', 'char_freq_['], axis=1)
print(df2)

# checking duplicates
df2.duplicated()

# dropping the duplicates which says 'True'
df2.drop_duplicates(inplace=True)
print(df2)

# Storing the values from start to 56th column in y array
y = df2.iloc[:, 55].values

# providing the summary of descriptive statistics count, mean, standard deviation, minimum and maximum values,
# as well as the 25th, 50th, and 75th percentiles of the data
df2.describe()

# standardizing the database
scaler = StandardScaler()
# applying standardscaler to the database
scaled_data = scaler.fit_transform(df2)

# creating new dataframe using the data from df2
df3 = pd.DataFrame(data=scaled_data, columns=df2.columns)
print(df3)

# providing the summary of descriptive statistics
df3.describe()

# creating an instance
pca = PCA()
# fitting to the dataframe
principalComponents = pca.fit_transform(df3)
# plotting the graph
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('Explained Variance Ratio')
plt.xlabel('Number Of Components')
plt.ylabel('Variance Ratio')
plt.show()

# transforming the original features of the dataset to a new set of features
pca = PCA(n_components=45)
new_data = pca.fit_transform(df3)

# new dataset
principal_Df = pd.DataFrame(data=new_data)
print(principal_Df)

np.random.seed(1)
X = np.dot(np.random.random(size=(45, 45)), np.random.normal(size=(45, 4208))).T
plt.plot(X[:, 0], X[:, 1], 'o')
plt.axis('equal');
new = pd.DataFrame(X)
print(new)



