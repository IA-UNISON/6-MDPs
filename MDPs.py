"""
Clases y funciones para definir y resolver MDPs discretos
utilizando programación dinámica y Q-learning en forma tabular.

"""
from abc import ABCMeta, abstractmethod
from random import choice, random

class MDP(metaclass=ABCMeta):
    """
    Clase para definir un MDP discreto.
    
    Es necesario establecer 
        - La forma de representar el estado como una tupla (s \in S)
        - La forma de representar las acciones (a \in A)
        - Un factor de descuento gama
        
    Los métodos que deben implementarse son:
        - acciones_legales(s)
        - recompensa(s, a, s')
        - prob_transicion(s, a, s')
        - es_terminal(s)
        
    """
    def __init__(self, estados, gama):
        self.estados = estados
        self.gama = gama
       
    @abstractmethod
    def acciones_legales(self, s):
        """
        Devuelve una lista con las acciones legales en el estado s.
        
        """
        raise NotImplementedError("Acciones legales no implementada")
    
    @abstractmethod
    def recompensa(self, s, a, s_):
        """
        Devuelve la recompensa de la transición s, a, s'.
        
        """
        raise NotImplementedError("Recompensa no implementada")
    
    @abstractmethod
    def prob_transicion(self, s, a, s_):
        """
        Devuelve la probabilidad de la transición s, a, s'.
        
        """
        raise NotImplementedError("Probabilidad de transición no implementada")
    
    @abstractmethod
    def es_terminal(self, s):
        """
        Devuelve True si el estado s es terminal.
        
        """
        raise NotImplementedError("Es terminal no implementada")
    

def valor_politica(pi, mdp, epsilon=1e-6, max_iter=1000):
    """
    Calcula el valor de una política pi para un MDP.
    
    Parámetros
    ----------
    pi : dict
        Política pi que asigna a cada estado una acción.
    mdp : MDP
        MDP para el que se calcula el valor de la política.
    epsilon : float
        Criterio de convergencia.
    max_iter : int
        Número máximo de iteraciones.
        
    Devuelve
    --------
    V : dict
        Valor de la política pi.
    
    """
    V = {s: 0 for s in mdp.estados}
    
    for _ in range(max_iter):
        delta = 0
        for s in mdp.estados: 
            if not mdp.es_terminal(s):
                v = V[s]
                V[s] = sum(
                    mdp.prob_transicion(s, pi[s], s_) 
                    * (mdp.recompensa(s, pi[s], s_) + mdp.gama * V[s_])
                    for s_ in mdp.estados
                )
                delta = max(delta, abs(v - V[s]))
        if delta < epsilon:
            break
    return V

def iteracion_politica(mdp, epsilon=1e-6, max_iter=1000):
    """
    Calcula la política óptima para un MDP utilizando iteración de política.
    
    Parámetros
    ----------
    mdp : MDP
        MDP para el que se calcula la política óptima.
    epsilon : float
        Criterio de convergencia.
    max_iter : int
        Número máximo de iteraciones.
        
    Devuelve
    --------
    pi : dict
        Política óptima.
    
    """
    pi = {s: choice(mdp.acciones_legales(s)) 
          for s in mdp.estados if not mdp.es_terminal(s)}
    
    for _ in range(max_iter):
        V = valor_politica(pi, mdp, epsilon, max_iter)
        optima = True
        for s in mdp.estados:
            if not mdp.es_terminal(s):
                a = pi[s]
                pi[s] = max(
                    mdp.acciones_legales(s),
                    key=lambda a: sum(
                        mdp.prob_transicion(s, a, s_) 
                        * (mdp.recompensa(s, a, s_) + mdp.gama * V[s_])
                    for s_ in mdp.estados
                    )
                )
                if a != pi[s]:
                    optima = False
            if optima:
                break

    return pi

def iteracion_valor(mdp, epsilon=1e-6, max_iter=1_000, debug=False):
    """
    Calcula la política óptima para un MDP utilizando iteración de valor.
    
    Parámetros
    ----------
    mdp : MDP
        MDP para el que se calcula la política óptima.
    epsilon : float
        Criterio de convergencia.
    max_iter : int
        Número máximo de iteraciones.
    debug : bool
        Si es True, imprime el valor de delta cada 100 iteraciones.
        
    Devuelve
    --------
    pi : dict
        Política óptima.
    
    """
    V = {s: 0 for s in mdp.estados}
    
    for it in range(max_iter):
        delta = 0
        for s in mdp.estados:
            if not mdp.es_terminal(s):
                v = V[s]
                V[s] = max(
                    sum(
                        mdp.prob_transicion(s, a, s_) 
                        * (mdp.recompensa(s, a, s_) + mdp.gama * V[s_])
                        for s_ in mdp.estados
                    )
                    for a in mdp.acciones_legales(s)
                )
                delta = max(delta, abs(v - V[s]))
        if debug and it % 50 == 0:
            print(f"Iteración {it + 1} - Delta: {delta}")
        if delta < epsilon:
            break
    
    pi = {s: max(
        mdp.acciones_legales(s),
        key=lambda a: sum(
            mdp.prob_transicion(s, a, s_) 
            * (mdp.recompensa(s, a, s_) + mdp.gama * V[s_])
            for s_ in mdp.estados
        )
    ) for s in mdp.estados if not mdp.es_terminal(s)}
    
    return pi, V
    
    