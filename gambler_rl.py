"""
El problema del jugador pero como un problema de aprendizaje por refuerzo

"""

from RL import MDPsim, SARSA, Q_learning
from random import random, randint

class Jugador(MDPsim):
    """
    Clase que representa un MDP para el problema del jugador.
    
    El jugador tiene un capital inicial y el objetivo es llegar a un capital
    objetivo o quedarse sin dinero.
    
    """
    def __init__(self, meta, ph, gama):
        self.estados = tuple(range(meta + 1))
        self.meta = meta
        self.ph = ph
        self.gama = gama
        
    def estado_inicial(self):
        return randint(1, self.meta - 1)
    
    def acciones_legales(self, s):
        if s == 0 or s == self.meta:
            return []
        return range(1, min(s, self.meta - s) + 1)
    
    def recompensa(self, s, a, s_):
        return 1 if s_ == self.meta else 0
    
    def transicion(self, s, a):
        return s + a if random() < self.ph else s - a
    
    def es_terminal(self, s):
        return s == 0 or s == self.meta
    
mdp_sim = Jugador(
    meta=100, ph=0.40, gama=1
)

Q_sarsa = SARSA(
    mdp_sim, 
    alfa=0.2, epsilon=0.02, n_ep=300_000, n_iter=50
)
pi_s = {s: max(
    mdp_sim.acciones_legales(s), key=lambda a: Q_sarsa[(s, a)]
) for s in mdp_sim.estados if not mdp_sim.es_terminal(s)}

Q_ql = Q_learning(
    mdp_sim, 
    alfa=0.2, epsilon=0.02, n_ep=300_000, n_iter=50
)
pi_q = {s: max(
    mdp_sim.acciones_legales(s), key=lambda a: Q_ql[(s, a)]
) for s in mdp_sim.estados if not mdp_sim.es_terminal(s)}

print("Estado".center(10) + '|' +  "SARSA".center(10) + '|' + "Q-learning".center(10))
print("-"*10 + '|' + "-"*10 + '|' + "-"*10)
for s in mdp_sim.estados:
    if not mdp_sim.es_terminal(s):
        print(str(s).center(10) + '|' 
              + str(pi_s[s]).center(10) + '|' 
              + str(pi_q[s]).center(10))
print("-"*10 + '|' + "-"*10 + '|' + "-"*10)

""" 
***************************************************************************************
Responde las siguientes preguntas:
***************************************************************************************
1. ¿Qué pasa si se modifica el valor de epsilón de la política epsilon-greedy?
cuando aumenta, se hacen mas recorridos y explora mas, caso contrario este explora menos, haciendo menos recorridos
y usa lo que se encontro

2. ¿Para que sirve usar una politica epsilon-greedy?
para explorar todos los posibles caminos que den a mejores recompensas que tiene a su alcanze 

3. ¿Qué pasa con la política óptima y porqué si p_h es mayor a 0.5?
la primera cosa que pasaria, es que se desnivela, y hay mas probabilidad de tener recompensas


4. ¿Y si es 0.5?
se nivelea

5. ¿Y si es menor a 0.5?
hay menor probabilidad de ganar y no hay recompensas

6. ¿Qué pasa si se modifica el valor de la tasa de aprendizaje?
puede variar segun si es alta o baja, y varia en su aprendizaje, puede ser rapido o mas lento

7. ¿Qué pasa si se modifica el valor de gama?
aplica lo mismo de recompensas futuras, y conserva mas si disminuye gama

***************************************************************************************

"""