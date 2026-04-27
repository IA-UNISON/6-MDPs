"""
El problema del jugador pero como un problema de aprendizaje por refuerzo

"""

from RL import MDPsim, SARSA, Q_learning, PoliticaGreedy
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
        return [i for i in range(1, min(s, self.meta - s) + 1)]
    
    def recompensa(self, s, a, s_):
        return self.meta if s_ == self.meta else 0
    
    def transicion(self, s, a):
        return s + a if random() < self.ph else s - a
    
    def es_terminal(self, s):
        return s == 0 or s == self.meta
    
mdl = Jugador(meta=100, ph=0.40, gama=1)

Q_sarsa = SARSA( mdl, alfa=0.2, epsilon=0.02, n_ep=10_000, n_iter=100)
pi_s = PoliticaGreedy(Q_sarsa)

Q_ql = Q_learning( mdl, alfa=0.2, epsilon=0.02, n_ep=10_000, n_iter=100)
pi_q = PoliticaGreedy(Q_ql)

print("Estado".center(10) + '|' +  "SARSA".center(10) + '|' + "Q-learning".center(10))
print("-"*10 + '|' + "-"*10 + '|' + "-"*10)
for s in mdl.estados:
    if not mdl.es_terminal(s):
        print(str(s).center(10) + '|' 
              + str(pi_s(s)).center(10) + '|' 
              + str(pi_q(s)).center(10))
print("-"*10 + '|' + "-"*10 + '|' + "-"*10)

""" 
***************************************************************************************
Responde las siguientes preguntas:
***************************************************************************************
1. ¿Qué pasa si se modifica el valor de epsilón de la política epsilon-greedy?
    Si aumentamos el epsilon, el agente va a explorar más, va a elegir más veces 
    acciones al azar, mientras que si lo disminuimos va a explorar menos y va a seguir
    uasndo de lo que ya aprendió. En cualquier caso, un epsilon más alto puede complicar
    encontrar una política óptima pues siempre estará buscando otras, mientras que un
    epsilon bajo, puede que tampoco encuentre la mejor política por no explorar
    lo suficiente.

2. ¿Para que sirve usar una politica epsilon-greedy?
    Para hacer un balance entre exploración y explotación, permitiendo que se pueben 
    diferentes acciones pero también que se tome la mejor acción posible en el momento

3. ¿Qué pasa con la política óptima y porqué si p_h es mayor a 0.5?
    si p_h es mayor a 0.5 hay una posibilidad más grande de ganar que de perder, por lo
    que la política más óptima sería apostar la máxima cantidad posible en cada jugada 
    para llegar rápido a la meta, porque matemáticamente es ventajoso arriesgar mucho.

4. ¿Y si es 0.5?
    Si es igual a 0.5 pues es tan justo como tirar una moneda, el resultado esperado no
    cambia sin importar la apuesta porque no hay una ventaja.

5. ¿Y si es menor a 0.5?
    Sería mejor no apostar o apostar lo mínimo ya que porque el juego está en tu contra.
    Mientras más se apueste, más rápido perderemos ganancia. Aquí la política óptima 
    sería apostar lo mínimo permitido.

6. ¿Qué pasa si se modifica el valor de la tasa de aprendizaje?
    El agente se adapta más rápido pero puede ser inestable. Si alfa es muy pequeño,
    aprende muy lento y puede tardar demasiado en mejorar su política

7. ¿Qué pasa si se modifica el valor de gama?
    Entre más bajo, más importancia se le da a la ganancia inmediata, si es más alto,
    se le dará más importancia a las recompensas a largo plazo.

***************************************************************************************

"""