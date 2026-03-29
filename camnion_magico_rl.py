"""
El camión mágico, pero ahora por simulación

"""

from RL import MDPsim, SARSA, Q_learning
from random import random, randint

class CamionMagico(MDPsim):
    """
    Clase que representa un MDP para el problema del camión mágico.
    
    Si caminas, avanzas 1 con coso 1
    Si usas el camion, con probabilidad rho avanzas el doble de donde estabas
    y con probabilidad 1-rho te quedas en el mismo lugar. Todo con costo 2.
    
    El objetivo es llegar a la meta en el menor costo posible
    
    """    
    
    def __init__(self, gama, rho, meta):
        self.gama = gama
        self.rho = rho
        self.meta = meta
        self.estados = tuple(range(1, meta + 2))
    
    def estado_inicial(self):
        #return randint(1, self.meta // 2 + 1)
        return randint(1, self.meta - 1)
    
    def acciones_legales(self, s):
        if s >= self.meta:
            return []
        return ['caminar', 'usar_camion']
    
    def recompensa(self, s, a, s_):
        return (
            -100  if s_ > self.meta else
             100  if s_ == self.meta else
            -1  if a == 'caminar' else -2   
        ) 
        
    def transicion(self, s, a):
        if a == 'caminar':
            return min(s + 1, self.meta + 1)
        elif a == 'usar_camion':
            return min(self.meta + 1, 2*s) if random() < self.rho else s
        
    def es_terminal(self, s):
        return s >= self.meta
def testing(roro:float = 0.99, gamma:float = 0.999, episodes:int = 100_000, SARSA_iter:int = 50, Q_iter:int = 1000):
    mdp_sim = CamionMagico(
        gama=gamma, rho=roro, meta=145
    )
        
    Q_sarsa = SARSA(
        mdp_sim, 
        alfa=0.1, epsilon=0.02, n_ep=episodes, n_iter=SARSA_iter
    )
    pi_s = {s: max(
        ['caminar', 'usar_camion'], key=lambda a: Q_sarsa[(s, a)]
    ) for s in mdp_sim.estados if not mdp_sim.es_terminal(s)}

    Q_ql = Q_learning(
        mdp_sim, 
        alfa=0.1, epsilon=0.02, n_ep=episodes, n_iter=Q_iter
    )
    pi_ql = {s: max(
        ['caminar', 'usar_camion'], key=lambda a: Q_ql[(s, a)]
    ) for s in mdp_sim.estados if not mdp_sim.es_terminal(s)}

    print(f"Los tramos donde se debe usar el camión segun SARSA son:")
    print([s for s in pi_s if pi_s[s] == 'usar_camion'])
    print("-"*50)
    print(f"Los tramos donde se debe usar el camión segun Qlearning son:")
    print([s for s in pi_ql if pi_ql[s] == 'usar_camion'])
    print("-"*50)


rhos = [0.4,0.5,0.6,0.7,0.8,0.9,0.999]
gammas = [0.5,0.6,0.7,0.8,0.9,0.999]
for _ in range (10):
    #for rho in rhos:
        #print(f"Para rho = {rho}")
        #testing(roro=rho)

    #for gamma in gammas:
        #print(f"Para gamma = {gamma}")
        #testing(gamma=gamma)
    testing(episodes=10_000)
"""
**********************************************************************************
Ahora responde a las siguientes preguntas:
**********************************************************************************

- Prueba con diferentes valores de rho. ¿Qué observas? ¿Porqué crees que pase eso?
- Prueba con diferentes valores de gama. ¿Qué observas? ¿Porqué crees que pase eso?
- ¿Qué tan diferente es la política óptima de SARSA y Q-learning?
- ¿Cambia mucho el resultado cambiando los valores de recompensa?
- ¿Cuantas iteraciones se necesitan para que funcionen correctamente los algoritmos?
- ¿Qué pasaria si ahora el estado inicial es cualquier estado de la mitad para abajo?
**********************************************************************************

"""
