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

mdp_sim = CamionMagico(
    gama=0.999, rho=0.3, meta=145
)
    
Q_sarsa = SARSA(
    mdp_sim, 
    alfa=0.1, epsilon=0.02, n_ep=100_000, n_iter=50
)
pi_s = {s: max(
    ['caminar', 'usar_camion'], key=lambda a: Q_sarsa[(s, a)]
) for s in mdp_sim.estados if not mdp_sim.es_terminal(s)}

Q_ql = Q_learning(
    mdp_sim, 
    alfa=0.1, epsilon=0.02, n_ep=100_000, n_iter=1000
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


"""
**********************************************************************************
Ahora responde a las siguientes preguntas:
**********************************************************************************

- Prueba con diferentes valores de rho. ¿Qué observas? ¿Porqué crees que pase eso?
    En este caso, el valor de rho representa la probabilidad de que el camión funcione
    y avances el doble.

    Si el valor de rho es muy alto, se utiliza mas el camion ya que la probabilidad de 
    avanzar el doble es mas alta. Aunque nos cuesta mas, la probailidad lo compensa.

    Si el valor de rho es muy bajo, se utiliza menos el camion ya que la probabilidad de
    avanzar el doble es baja y no nos combiene arriesgarnos ya que el costo de usar el 
    camion es mas alto.

    Si probamos valores:

    Para rho > 0.5:
    - El camión funnciona mas, por lo que el agente se arriesga mas
    Para rho < 0.5:
    - El camión funciona menos, por lo que el agente se arriesga menos
    Para rho = 0.5:
    - El camión tiene una probabilidad mas justa, es una estrategia viable pero arriesgada.

- Prueba con diferentes valores de gama. ¿Qué observas? ¿Porqué crees que pase eso?

    El valor de gamma representa la importancia que tienen las recompensas futuras.
    Si el valor de gamma es muy alto, el agente se concentra en su ganacia futura.
    Si el valor de gamma es muy bajo, el agente se concentra en ganacias inmediatas.

    para gammas < .3:
    - El agente quiere estar caminando mas seguido, ya que le atrae el costo bajo de caminar
    para gammas > .3 y gammas < .7:
    - El agente es mas neutro, de vez en cuado se arriesga a tomar el camion
    para gammas > .7:
    - El agente valora mas la ganacia futura, usa mas seguido el camion ya que esto lo
    puede llevar a un mejor estado, aunque el costo inmediato sea mas alto.


- ¿Qué tan diferente es la política óptima de SARSA y Q-learning?
    Q-learning tiende a ser más optimista, siempre actualiza segun la mejor
    acción posible, aunque no se esté tomando.

    SARSA aprende según las acciones que realmente toma, por lo que puede ser
    más conservador.

- ¿Cambia mucho el resultado cambiando los valores de recompensa?
    Si y bastante, ya que los valores de la recompensa influyen direcatmente en las politicas.

    Por ejemplo:
    - Si aumentaramos la penalizacion por usar el camion, el agente sería mas cauteloso al 
    momento de usar el camion.

- ¿Cuantas iteraciones se necesitan para que funcionen correctamente los algoritmos?
    En estos tipos de probelmas, el numero de iteraciones depende de varias factores,
    pero en general si es bueno usar una cantidad de iteraciones altas, ya que nuestro
    numero es bajo, podriamos tener resultados no tan optimos. Para el ejercicio estamos
    usando 100,000 iteraciones, lo cual es un numero bueno, pero si consume mas tiempo 
    de procesamiento.

- ¿Qué pasaria si ahora el estado inicial es cualquier estado de la mitad para abajo?
    A lo que entiendo, si el estado inicial es cualquier estado desde 1 hasta meta/2, 
    el camion se vuelve mas atractivo, ya que podemos avanzar mas y mas rapido.
**********************************************************************************

"""