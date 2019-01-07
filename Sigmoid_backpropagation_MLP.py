
import numpy as np

weightI = np.array([[0.2,0.7],
                   [-0.1,-1.2]]) #inizializzo i pesi per 3 unità input

weightH = np.array([[1.1,3.1],
                  [0.1,1.17]]) #inizializzo pesi per 2 unità nascoste

def output_pesati(input,pesi):  # funzione che mi calcola gli output pesati in un nodo con i valori di pesi e input
  sum=0
  for x , w in zip(input,pesi):
    sum = sum + x*w
  return sum

def sigmoide(x): # calcola la sigmoide del valore dato in input
  return 1/(1+np.exp(-x))

def output(training,weightI,weightH):  #calcola l'output della rete neurale con dati di training e pesi per nodi di input e nodi hidden
  H1 = sigmoide(output_pesati(training[0:(np.size(training))-1],weightI[:,0]))
  H2 = sigmoide(output_pesati(training[0:(np.size(training))-1],weightI[:,1]))
  arrayH = np.append(H1,H2)
  O1 = sigmoide(output_pesati(arrayH,weightH[:,0]))
  O2 = sigmoide(output_pesati(arrayH,weightH[:,1]))
  return np.append(O1,O2)

def deltaO(arrayO,training):   # calcola il valore di delta (delta k) su output cioè zk(1−zk)(tk −zk)   
  errorO1 = arrayO[0]*(1-arrayO[0])*(training[(np.size(training))-1]-arrayO[0])
  errorO2 = arrayO[1]*(1-arrayO[1])*(1-training[(np.size(training))-1]-arrayO[1])
  return np.append(errorO1,errorO2)

def hidden():  #calcola il valore uscente dai nodi hidden, pesati e applicato la funzione sigmoide sul valore uscente 
  H1 = sigmoide(output_pesati(training[0:(np.size(training))-1],weightI[:,0]))
  H2 = sigmoide(output_pesati(training[0:(np.size(training))-1],weightI[:,1]))
  return np.append(H1,H2)

def deltaH(deltaO): #calcola delta (delta j) dei nodi hidden
  errorH1 = output_pesati(deltaO,weightH[0,:])*hidden()[0]*(1-hidden()[0])
  errorH2 = output_pesati(deltaO,weightH[1,:])*hidden()[1]*(1-hidden()[1])
  return np.append(errorH1,errorH2)

def aggiornapesiI(deltaH,OldPesiI,eta,training): #aggiorna i pesi degli input sui nodi hidden cioè Wji
  NewPesiI = OldPesiI
  i=0
  for w in NewPesiI:
    w[0]=w[0]+eta*deltaH[0]*training[i]
    w[1]=w[1]+eta*deltaH[1]*training[i]
    i=i+1
  return NewPesiI

def aggiornapesiO(deltaO,OldPesiH,eta,training):  #aggiorna i pesi degli output dei nodi hidden sui nodi output cioè Wkj
  NewPesiO = OldPesiH
  i=0
  for w in NewPesiO:
    w[0]=w[0]+eta*deltaO[0]*training[i]
    w[1]=w[1]+eta*deltaO[1]*training[i]
    i=i+1
  return NewPesiO

def aggiornaPesi(deltaH,deltaO,OldPesiH,OldPesiI,eta,training):  #ritorna i pesi aggiornati
  NewPesiI = aggiornapesiI(deltaH,OldPesiI,eta,training)

  NewPesiO = aggiornapesiO(deltaO,OldPesiH,eta,training)

  return np.append(NewPesiI,NewPesiO)

training=np.array([[      
2.7810836	,	2.550537003	,	0],
[1.465489372	,	2.362125076	,	0],
[3.396561688	,	4.400293529	,	0],
[1.38807019	,	1.850220317	,	0],
[3.06407232	,	3.005305973	,	0],
[7.627531214	,	2.759262235	,	1],
[5.332441248	,	2.088626775,		1],
[6.922596716	,	1.77106367	,	1],

    [7.673756466	,	3.508563011	,	1]
])

# Alleno la rete neurale


validation=np.zeros(0)   #inteso O2 = 0     e O1 = 1  cioè  se O2>O1 allora etichetta 0  altrimenti se O1>O2 allora etichetta 1
i=0
while i<10: #utilizzo 10 iterazioni
  for t in training: #per ogni elemento da classificare calcolo l'output con i pesi correnti
    O1 = output(t,weightI,weightH)[0]
    O2 = output(t,weightI,weightH)[1]
    
    if O2 > O1:  #controllo se l'elemento è stato classificato correttamente
      
      if(t[2]==1): # allora è missclassificato quindi aggiorno i pesi
        
        i=i-1
        PesiAggiornati = aggiornaPesi(deltaH(deltaO(output(t,weightI,weightH),t)),deltaO(output(t,weightI,weightH),t),weightH,weightI,0.3,t) #aggiorna tutti i pesi
      else:
        
        i=i+1
      validation = np.append(validation,O2)
    else:
      
      if(t[2]==0): # allora è missclassificato quindi aggiorno i pesi
   
        i=i-1
        PesiAggiornati = aggiornaPesi(deltaH(deltaO(output(t,weightI,weightH),t)),deltaO(output(t,weightI,weightH),t),weightH,weightI,0.3,t) #aggiorna tutti i pesi
      else:
        
        i=i+1
      validation = np.append(validation,O1)

# calcolo i risultati della rete neurale

validation=np.zeros(0)
for t in training:
  O1 = output(t,weightI,weightH)[0]
  O2 = output(t,weightI,weightH)[1]
  print(O1,O2)
  if O2 > O1:
    validation = np.append(validation,0)
  else:
    validation = np.append(validation,1)
print(validation)

# valido la rete neurale su un elemento del validation set

O1 = output(np.array([8.675418651	,	-0.242068655	,	1]),weightI,weightH)[0]
O2 = output(np.array([8.675418651	,	-0.242068655	,	1]),weightI,weightH)[1]
print(O1,O2)

