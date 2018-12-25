

import pandas as pd;

from sklearn.tree import DecisionTreeClassifier;

! git clone --recursive https://github.com/selva86/datasets.git;

!ls

dataset = pd.read_csv('datasets/Zoo.csv')

train_items = dataset.iloc[:80,:-1]; #prendi le prime 80 righe e tutte le colonne tranne l'ultima

test_items = dataset.iloc[80:,:-1]

train_target = dataset.iloc[:80,-1] #prendo per tutte le prime 80 righe del training set il target

test_target = dataset.iloc[80:,-1] #prendo i rimanenti target

ID3 = DecisionTreeClassifier(criterion = 'entropy').fit(train_items,train_target) #alleno il modello

predictions = ID3.predict(test_items);

print(predictions)

print(dataset.iloc[80:,-1])

print("The prediction accuracy is: ",ID3.score(test_items,test_target)*100,"%")

# DA QUI UTILIZZO IL DATASET ADULT.CSV

dataset2 = pd.read_csv('datasets/adultTrain.csv');

for i,item in enumerate(dataset2.iloc[:,0]): #rendo discreti i valori dell'età
  if item<30:
    dataset2.at[i,"age"] = 0; 
  if item>=30 and item<50:
    dataset2.at[i,"age"] = 1; 
  if item>=50:
    dataset2.at[i,"age"] = 2;

print(dataset2.iloc[:100,0])

print(dataset2["workclass"].value_counts()); #capisco quali sono i valori da sistemare

from sklearn.preprocessing import LabelEncoder;

lb_make = LabelEncoder();

dataset2["workclass"]=lb_make.fit_transform(dataset2["workclass"]); #trasformo in numeri il lavoro

print(dataset2.iloc[:100,1]);

train_items2 = dataset2.iloc[:8000,:2]; #prendi le prime 8000 righe e tutte le colonne tranne l'ultima

test_items2 = dataset2.iloc[8000:,:2] #prendo le altre righe e tutte le colonne tranne l'ultima

train_target2 = dataset2.iloc[:8000,-1] #prendo per tutte le prime 8000 righe del training set il target

test_target2 = dataset2.iloc[8000:,-1] #prendo i rimanenti target

ID32 = DecisionTreeClassifier(criterion = 'entropy').fit(train_items2,train_target2) #alleno il modello

print(test_target2);

predictions = ID32.predict(test_items2);

print("The prediction accuracy is: ",ID32.score(test_items2,test_target2)*100,"%")

print(dataset2["fnlwgt"].value_counts()); #capisco quali sono i valori da sistemare

dataset2 = dataset2.drop('fnlwgt', 1); #elimino la colonna "fnlwgt" poichè inutile ai fini della classificazione

print(dataset2["education"].value_counts()); #occorrenze diverse del grado scolastico

dataset2["education"]=lb_make.fit_transform(dataset2["education"]); #trasformo in numeri il grado scolastico

print(dataset2.iloc[:100,2]);

print(dataset2["education_num"].value_counts()); #occorrenze diverse del grado scolastico

dataset2 = dataset2.drop('education_num', 1); #elimino la colonna "educazion_num" poichè duplicata, ai fini della classificazione

print(dataset2["marital_status"].value_counts()); #occorrenze diverse dello stato civile

dataset2["marital_status"]=lb_make.fit_transform(dataset2["marital_status"]); #trasformo in numeri lo stato civile

print(dataset2.iloc[:100,3]);

print(dataset2["occupation"].value_counts()); #occorrenze diverse del lavoro svolto

dataset2["occupation"]=lb_make.fit_transform(dataset2["occupation"]); #trasformo in numeri il lavoro

print(dataset2["relationship"].value_counts()); #occorrenze diverse dello stato famiglia

dataset2["relationship"]=lb_make.fit_transform(dataset2["relationship"]); #trasformo in numeri lo stato famiglia

print(dataset2["race"].value_counts()); #occorrenze diverse  della provenienza

dataset2["race"]=lb_make.fit_transform(dataset2["race"]); #trasformo in numeri il paese d origine

print(dataset2["sex"].value_counts()); #occorrenze diverse  del sesso

dataset2["sex"]=lb_make.fit_transform(dataset2["sex"]); #trasformo in numeri il sesso

print(dataset2["capital_gain"].value_counts()); #occorrenze diverse  dei guadagni

for i,item in enumerate(dataset2["capital_gain"]): #rendo discreti i valori del capitale guadagnato
  if item>6000 and item<12000:
    dataset2.at[i,"capital_gain"] = 0; 
  if item<=6000 :
    dataset2.at[i,"capital_gain"] = 1; 
  if item>=12000:
    dataset2.at[i,"capital_gain"] = 2;

print(dataset2["capital_loss"].value_counts()); #occorrenze diverse  delle perdite

for i,item in enumerate(dataset2["capital_loss"]): #rendo discreti i valori del capitale perso
  if item==0:
    dataset2.at[i,"capital_loss"] = 0; 
  else:
    dataset2.at[i,"capital_loss"] = 1;

print(dataset2["hours_per_week"].value_counts()); #occorrenze diverse  delle ore lavorate

for i,item in enumerate(dataset2["hours_per_week"]): #rendo discreti i valori delle ore lavorate
  if item>10 and item<25:
    dataset2.at[i,"hours_per_week"] = 1; 
  if item<=10 :
    dataset2.at[i,"hours_per_week"] = 0; 
  if item>=25 and item<40:
    dataset2.at[i,"hours_per_week"] = 2; 
  if item>=40:
    dataset2.at[i,"hours_per_week"] = 3;

print(dataset2["native_country"].value_counts()); #occorrenze diverse  delle nazioni di provenienza

dataset2 = dataset2.drop('native_country', 1); #elimino la colonna "native_country" poichè preferisco tenere "race" che è piu generale

train_items3 = dataset2.iloc[:8000,:-1]; #prendi le prime 8000 righe e tutte le colonne tranne l'ultima
test_items3 = dataset2.iloc[8000:,:-1]; #prendo le altre righe e tutte le colonne tranne l'ultima
train_target3 = dataset2.iloc[:8000,-1]; #prendo per tutte le prime 8000 righe del training set il target
test_target3 = dataset2.iloc[8000:,-1]; #prendo i rimanenti target

ID33 = DecisionTreeClassifier(criterion = 'entropy').fit(train_items3,train_target3) #alleno il modello

predictions = ID33.predict(test_items3);

print("The prediction accuracy is: ",ID33.score(test_items3,test_target3)*100,"%")

