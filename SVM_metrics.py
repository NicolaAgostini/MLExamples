
import numpy as np;

np.random.seed(10);
X_xor = np.random.randn(200,2); #genero matrice con 200 valori random dalla distribuzione normale standard

y_xor = np.logical_xor(X_xor[:,0] > 0 , X_xor[:,1] > 0); #genero i corrispondenti valori y dati dallo xor tra elementi delle due colonne di X_xor

y_xor = np.where(y_xor,1,-1); #trasformo i valori in -1 (false) e 1 (true)

import matplotlib.pyplot as plt;

plt.scatter(X_xor[y_xor==1,0], X_xor[y_xor==1,1], c="orange", marker="^",label="true"); #considero come verdi i campioni di addestramento in cui il risultato dello xor è = 1
plt.scatter(X_xor[y_xor==-1,0], X_xor[y_xor==-1,1],c="b", marker="s",label="false");#considero come verdi i campioni di addestramento in cui il risultato dello xor è = -1
plt.legend();
plt.show();

from sklearn.svm import SVC;

svm = SVC(kernel="rbf", gamma=0.3, C=1.0);

from sklearn.model_selection import train_test_split;

X_train, X_test, y_train, y_test = train_test_split(X_xor, y_xor, test_size = 0.30);



svm.fit(X_train, y_train);

y_preditions = svm.predict(X_test);



from mlxtend.plotting import plot_decision_regions;

plot_decision_regions(X_train, y_train, clf=svm);
plt.legend(loc="upper right");
plt.show();

from sklearn.metrics import classification_report, confusion_matrix; #importo le metriche precision recall etc..

print(confusion_matrix(y_test,y_preditions)); #stampo la confusion metrics

print(classification_report(y_test,y_preditions)) ;

from google.colab import files
uploaded = files.upload()

!ls

import pandas as pd;

data_set = pd.read_csv("bill_authentication.csv"); #import un nuovo dataset

X = data_set.drop('Class', axis=1); #eseguo preprocessing eliminando dai campioni di addestramento il target

y = data_set['Class']; #il vettore y rappresenta l'etichetta dei vari campioni di addestramento

from sklearn.model_selection import train_test_split;

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size = 0.30);

svm2 = SVC(kernel="linear", gamma=0.3, C=1.0);

svm2.fit(X_train2, y_train2);

y_preditions2 = svm2.predict(X_test2);

from sklearn.metrics import classification_report, confusion_matrix; #importo le metriche precision recall etc..

print(confusion_matrix(y_test2,y_preditions2)); #stampo la confusion metrics

print(classification_report(y_test2,y_preditions2)) ;

