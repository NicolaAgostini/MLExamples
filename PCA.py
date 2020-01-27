import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix



iris = load_iris()

data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

X = data1.drop("target", axis=1)
y = data1["target"] # classes





scaler = StandardScaler()  # mean 0 and variance 1
X = scaler.fit_transform(X)




# try PCA leaving out 2 dimensions
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
new_X = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])  # new data are from PCA transformation


print(new_X.head())


X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size = 0.20)



# see the points separated

a = new_X.plot(kind='scatter', x='PC1', y='PC2')

plt.show()



# try to classify new data obtained using PCA


svm = SVC(kernel='rbf')

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))





"""
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # load dataset in a panda dataframe
y = pd.Categorical.from_codes(iris.target, iris.target_names)  # load the label of each row of X

train, test = train_test_split(df, test_size=0.2)




print(X.head())


scaler = StandardScaler()  # mean 0 and variance 1
X = scaler.fit_transform(X)

print(X[1])

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
new_X = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])  # new data are from PCA transformation


print(new_X.head())


# see the points separated

a = new_X.plot(kind='scatter', x='PC1', y='PC2')

plt.show()


# try to classify new data obtained using PCA

clf = SVC(gamma='auto')

clf.fit(X, y)

"""





