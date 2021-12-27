import re
import numpy as np
import torch
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
        
        self.format = int(np.sum(self.transitions))
        cpt = 1
        for i in range(self.transitions.shape[0]):
            for j in range(self.transitions.shape[1]):
                if self.transitions[i][j]:
                    self.transitions[i][j] = cpt
                    cpt +=1
        

    def reset(self):
        self.current_state = self.start

    def id_state(self, s):
        return list(self.states).index(s)

    def one_hot(self, s1, s2):
        tmp = torch.zeros(self.format)
        tmp[int(self.transitions[s1][s2])-1] = 1
        return tmp

    def actions_possibles(self):
        id_state = self.id_state(self.current_state)
        return self.states[np.bool8(self.transitions[id_state])]


    def id_states_possibles_from(self,s):
        if type(s) is not int:
            s = self.id_state(s)
        return np.where(np.bool8(self.transitions[s]))[0]

    def action(self, act):
        """
        prends un état(str) ou un nombre représentant directement le xième état possible
        Renvoi l'état courant et le reward associé
        """
        if type(act) is int:
            act = self.actions_possibles()[act]

        if act in self.actions_possibles():
            self.current_state = act
            return act, self.rewards[self.states == act]
        raise "Value error : état non atteignable"

    def id_states_possibles_from(self,s):
        if type(s) is not int:
            s = self.id_state(s)
        return np.where(np.bool8(self.transitions[s]))[0]

if __name__ == "__main__":
    MC = Markov_chain('env0')

    print("ok")