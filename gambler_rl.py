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

    En este caso, el valor de epsilon representa la probabilidad de explorar
    una accion aleatoria, en lugar de escoger la mejor accion conocida al momento.

    Esto quiere decir que mientras mas alto sea el valor de epsilon, la probabilidad
    de esocger una accion aleatroria es mayor. Esto nos sirve para que el algoritmo
    explore nuevas estrategias.

    En el caso contrario, elegir un valor epsilon bajo hace que el algoritmo eliga con
    mas probailidad la mejor accion, lo que hace que el algoritmo explore menos pero es mas
    seguro.

2. ¿Para que sirve usar una politica epsilon-greedy?

    Precisamente usar una politica epsilon-greedy nos ayuda a balancear que tanto exploramos
    o que tanto usemos el conocimiento aprendido (Explotar)

3. ¿Qué pasa con la política óptima y porqué si p_h es mayor a 0.5?

    Cuando p_h es mayor a 0.5, esto siginifica que las probabilidades de ganar en el juego 
    son mayores a las de perder. Esto causaria que la politica apostará mas, con la
    intencion de ganar mas y asi llegar a la meta.

4. ¿Y si es 0.5?

    En este caso, la probabilidad de ganar es igual a la de perder. Esto significa que
    la politica tenderá a hacer apuestas mas pequeñas, con la intencion de reducir el 
    riesgo de perder. En este estado no hay una ventaja por apostar mas.

5. ¿Y si es menor a 0.5?
    En este caso, la probabilidad de ganar es menor a la de perder. Esto significa que
    la politica buscará apostar los menos posible, esto con la intencion de minimizar 
    daños. Se vuelve muy dificil ganar.

6. ¿Qué pasa si se modifica el valor de la tasa de aprendizaje?
    Si el valor de la tasa de aprendizaje es muy alta, el agente aprenderá rápido, pero
    es mas suceptible a cometer errores.
    Si el valor de la tasa de aprendizaje es muy bajo, el agente aprenderá lentamente,
    pero es mas seguro.

7. ¿Qué pasa si se modifica el valor de gama?

    El valor de gamma representa la importancia que tienen las recompenasas futuras.
    Si el valor de gamma es muy alto, el agente se concentra en su ganacia futura.
    Si el valor de gamma es muy bajo, el agente se concentra en ganacias inmediatas,
    lo que puede causar comportamientos erraticos..

***************************************************************************************

"""