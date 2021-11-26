import re
import numpy as np

class Markov_chain:

    def __init__(self,path):
        """
        Construit la chaine de markov présent dans le dossier path
        Toutes les lignes sont de la forme:
            start : nom_état_start
            Nom_état(val_need1, val_need2, ...)  :  Suc_1_nom, Suc_2_nom, ...
        """
        f = open(path).read()
        self.start = re.findall(r"^start\s*:\s*(.*)", f)[0]
        val   = np.array(re.findall("(?P<state>^.+)\((?P<needs>.+)\) .*\:(?P<link>.+)", f, re.M))
        self.rewards = np.array([np.float32(i.split(',')) for i in val[:,1]])#différents rewards par états
        self.states = val[:,0] # états dans l'ordre des lignes
        self.transitions = np.zeros((len(self.states), len(self.states))) #Ligne = départ, colonnes = arrivés
        for i,o in enumerate(val[:,2]):
            self.transitions[i] = np.int32(np.isin(self.states, o.split()))

if __name__ == "__main__":
    MC = Markov_chain('env0')

    print("ok")