import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# reading the file
df1 = pd.read_csv('spambase/spambase.data')
print(df1)
print()

with open('spambase/spambase.names') as spam:
    text = spam.read()
# finding the pattern
labels = re.findall(r'\n(\w*_?\W?):', text)
SpamData = pd.read_csv('spambase/spambase.data', header=None, names=labels +['spam'])
print(SpamData)
print()

# counting the number of boolean values and creating a boolean dataframe and adding up the 'true' values
SpamData_missing = SpamData.isna().sum()
print(SpamData_missing)
print()

df2 = pd.DataFrame(SpamData)
print(df2)
print()

# dropping unnecessary columns
df2 = df2.drop(labels=['word_freq_3d', 'char_freq_['],axis=1)
print(df2)
print()

# checking duplicates
print(df2.duplicated())
print()

# dropping the duplicates which says 'True'
df2.drop_duplicates(inplace=True)
print(df2)
print()

# Storing the all values of 56th column in y array
y = df2.iloc[:, 55].values

# providing the summary of descriptive statistics count, mean, standard deviation, minimum and maximum values,as well as the 25th, 50th, and 75th percentiles of the data
df2_description = df2.describe()
print(df2_description)
print()

# standardizing the features by removing the mean and scaling to unit variance
scaler = StandardScaler()
# applying standardscaler to the database
scaled_data = scaler.fit_transform(df2)

# creating new dataframe using the data from df2
df3 = pd.DataFrame(data=scaled_data, columns=df2.columns)
print(df3)
print()

# providing the summary of descriptive statistics
df3_description = df3.describe()
print(df3_description)
print()

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
print()

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
print()

# PCA model will hold 45 components which can be transformed the original data into a lower-dimensional space
pca = PCA(n_components=45)
pca.fit(y)
# getting the variance explained by each component
print(pca.explained_variance_)
print()
# returning the principal axes in feature space
print(pca.components_)
print()

# creating a scatter plot of the 1st and 2nd principal components
plt.plot(y[:, 0], y[:, 1], 'ok')

# loops through the eigenvalues and eigenvectors - for loop
# returning eigenvalues and eigenvectors
# converting these arrays into tuples - zip
for length, vector in zip(pca.explained_variance_, pca.components_):
    # length = eigenvalue of the principal component
    v = vector * 3 * np.sqrt(length)
    plt.plot([0, v[0]], [0, v[1]], 'y', lw=5)
plt.axis('equal')
plt.show()

# retaining 90% of the variance
npck = PCA(0.95)  # npck = number of principal components to keep
# fitting the PCA model to the data 'y' and transforms it to new set of principal components
y_fit_tran = npck.fit_transform(y)

# printing the shape of the y array: number of rows and columns
print('Shape of the numpy array "y"')
print(y.shape)
print(y_fit_tran.shape)
print()

# transforming compressed data back to the original feature
y_new = npck.inverse_transform(y_fit_tran)

plt.plot(y[:, 0], y[:, 1], 'ok', alpha=0.5)
plt.plot(y_new[:, 0], y_new[:, 1], 'ok', alpha=0.8)
plt.axis('equal')
plt.show()

print('array --> ')
print(y_fit_tran)
print()

df4 = pd.DataFrame(y_fit_tran)
print(df4)
print()
