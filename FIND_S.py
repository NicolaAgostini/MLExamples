
import numpy as np;

class Find_S:
  def __init__(self,training_example,target,n_dim):
    self.training_example = training_example;
    self.target = target;
    self.h=[];
    for i in range(0,n_dim):
      self.h.append("0"); #inizializzo h con l'ipotesi più specifica
    
  
  def output(self):
    for index,d in enumerate(self.training_example): #per ogni istanza di apprendimento
     
     if self.target[index] == "yes":  #se d è positivo
        
      for index,n_h in enumerate(self.h): #per ogni elemento di h
        if n_h != d[index]: #se l'elemento di h è diverso da l'istanza di apprendimento
          if n_h == "0": # se ho 0 all'inizio
            self.h[index] = d[index];
          else: # se ho un altro valore meno specifico rispetto a 0
            self.h[index] = "?"; #metto il valore più generale
    print(self.h);

def main():
  e1 = np.array([["big","red","circle"],
                 ["small","red","triangle"],
                 ["small","red","circle"],
                 ["big","blue","circle"],
                 ["small","blue","circle"]]);   #primo esempio
  
  e2 = np.array([['Sunny','Warm','Normal','Strong','Warm','Same'],
                 ['Sunny','Warm','High','Strong','Warm','Same'],
                 ['Rainy','Cold','High','Strong','Warm','Change'],
                 ['Sunny','Warm','High','Strong','Cool','Change']]);  #secondo esempio
  
  o1 = np.array(["no","no","yes","no","yes"]); #primo vettore di concetti target
  o2 = np.array(["yes","yes","no","yes"]); #secondo vettore di concetti target
  x=Find_S(e1,o1,3);
  x.output();
  
  x=Find_S(e2,o2,6);
  x.output();
  
  
if __name__ == '__main__':
    main();

