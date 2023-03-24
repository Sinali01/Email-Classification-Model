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

# providing the summary of descriptive statistics count, mean, standard deviation, minimum and maximum values,as well as the 25th, 50th, and 75th percentiles of the data
df2_description = df2.describe()
print(df2_description)

# standardizing the features by removing the mean and scaling to unit variance
scaler = StandardScaler()
# applying standardscaler to the database
scaled_data = scaler.fit_transform(df2)

# creating new dataframe using the data from df2
df3 = pd.DataFrame(data=scaled_data, columns=df2.columns)
print(df3)

# providing the summary of descriptive statistics
df3_description = df3.describe()
print(df3_description)

# applying Principal Component Analysis(PCA)
pca = PCA()
# fitting to the dataframe
pca_fit = pca.fit_transform(df3)

# plotting the PCA
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('Explained Variance Ratio')
plt.xlabel('Number Of Components')
plt.ylabel('Variance Ratio')
plt.show()

# transforming the original features of the dataset to a new set of features
pca = PCA(n_components=45)
new_data = pca.fit_transform(df3)

# assigning new data to a new dataframe to use for the algorithm
df_afterPCA = pd.DataFrame(data=new_data)
print(df_afterPCA)

# setting the random seed to 1
rng = np.random.default_rng(seed=1)

# creating an array y, multiplying two matrices and converting it to transpose array
y = np.dot(np.random.random(size=(45, 45)), np.random.normal(size=(45, 4208))).T

# plotting the values of first two principal components in a scatter plot
plt.plot(y[:, 0], y[:, 1], 'o')
# ensuring that x and y axes have equal scales
plt.axis('equal')
plt.show()

# assigning y into a new dataframe new_y
new_y = pd.DataFrame(y)
print(new_y)

# PCA model will hold 45 components which can be transformed the original data into a lower-dimensional space
pca = PCA(n_components=45)
pca.fit(y)
# getting the variance explained by each component
print(pca.explained_variance_)
# returning the principal axes in feature space
print(pca.components_)
