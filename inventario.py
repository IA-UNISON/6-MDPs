"""
Para desarrollar el problema del inventario.

"""

from MDPs import MDP, iteracion_valor
import math
import numpy as np

class Inventario(MDP):
    """
    Clase que representa un MDP para el problema del camión mágico.

    Quiero suponer que este problema era el de inventario de el ejercicio teórico
    https://ia-unison.github.io/2015/01/05/continua5.html
    Ya que en esta misma tarea ya se realiza el problema del camión mágico
    
    Si caminas, avanzas 1 con coso 1
    Si usas el camion, con probabilidad rho avanzas el doble de donde estabas
    y con probabilidad 1-rho te quedas en el mismo lugar. Todo con costo 2.
    
    El objetivo es llegar a la meta en el menor costo posible
    
    """    
    
    def __init__(self, gamma = 0.95,lambda_ = 4):
        self.gama = gamma
        self.lambda_ = lambda_
        """ 
        Estados (-15, -14, ..., 19, 20), 20 es el máximo de capacidad del almacen, 15 es estadisticamente
        imposible que ocurra en una porbabilidad dada por Poisson(k, 4), por lo cual lo establecemos como
        limite inferior
        """
        self.estados = tuple(range(-15, 21))
        self.s = 0
    
    def acciones_legales(self, s):
        if s >= 0:
            # 21 ya que range 0 a 20 no incluye a 20
            return range(0, 21 - s)
        else:
            return range(0, 21)

    """
    #---------------------------------------------------------------------------------------#
                                Método auxiliar para la recompensa
    #---------------------------------------------------------------------------------------#
    """
    #@staticmethod
    def poisson(self, k, lambd):
        return ((math.exp(-lambd)) * lambd**k ) / (math.factorial(k))

    def recompensa(self, s, a, s_):
        D = s + a - s_
        return ((150 * min(max(0, s+a), D)) 
                - ((80 * a) + (40 if a > 0 else 0))
                - (5 * max(0, s_)) 
                - (15 * max(0, -s_)) 
                - (70 * max(0, D - max(0, s + a)))
                )

    def prob_transicion(self, s, a, s_):
        D = s + a - s_

        # No es posible trener demanda negativa
        if D < 0:
            return 0.0
        
        # Límite inferior
        if s_ == -15:
            prob_acumulada_menor = sum(self.poisson(d, self.lambda_) for d in range(D))
            return 1.0 - prob_acumulada_menor
        
        # Caso normal
        return self.poisson(D, self.lambda_)

    def es_terminal(self, s):
        return False
    

if __name__ == "__main__":

    inventario = Inventario(0.9, 8)

    pi_star, V = iteracion_valor(inventario)

    print("-" * 68)
    print("Estado".center(20) + "Acción".center(20) + "Potencial de ganancia".center(20))
    print("-" * 68 )
    for s in pi_star:
        print(f"{s:^20}{pi_star[s]:^20}{V[s]:^20.2f}")
    print("-" * 68)


"""
    Contesta las preguntas aquí mismo (has espacio entre las preguntas):

    1. ¿Cómo se comporta las transiciones y las ganancias para casos específicos de $s$ y $a$?
    Mientras menor es s, mayor es la acción que se debe tomar para llegar a tener un retorno 

    2. ¿Qué pasa si hay mucho almacen? 
    La acción optima es no pedir nada, ya que solo tienes ventas y la demanda suele ser menor
    al inventario en almacén

    3. ¿Que pasa si hay muy poco o estamos sin almacen? 
    La acción optima es pedir entre 3 y 8, se pide para no caer en backlog

    4. ¿Existe un punto donde la ganancia sea máxima?  
    En este caso, al tener una demanda dada por una distribución de poisson y tener límite de 
    inventario = 20, el punto máximo es 20

    5. ¿Cómo se ve la política óptima? ¿Tiene sentido?
    --------------------------------------------------------------------
            -15                  20                601.30       
            -14                  20                720.23       
            -13                  20                815.46       
            -12                  20                898.64       
            -11                  19                978.64       
            -10                  18               1058.64       
            -9                  17               1138.64       
            -8                  16               1218.64       
            -7                  15               1298.64       
            -6                  14               1378.64       
            -5                  13               1458.64       
            -4                  12               1538.64       
            -3                  11               1618.64       
            -2                  10               1698.64       
            -1                  9                1778.64       
            0                   8                1858.64       
            1                   7                1938.64       
            2                   6                2018.64       
            3                   5                2098.64       
            4                   4                2178.64       
            5                   3                2258.64       
            6                   0                2360.23       
            7                   0                2455.46       
            8                   0                2538.64       
            9                   0                2616.75       
            10                  0                2692.40       
            11                  0                2765.71       
            12                  0                2836.15       
            13                  0                2903.43       
            14                  0                2967.57       
            15                  0                3028.73       
            16                  0                3087.04       
            17                  0                3142.59       
            18                  0                3195.44       
            19                  0                3245.65       
            20                  0                3293.29       
    --------------------------------------------------------------------
    Si, tiene todo el sentido, ya que si s es negativo, tenemos que estar pagando, por lo que
    necesitamos salir lo más pronto de ese estado, cuando tenemos poco inventario en almacen
    se pide poco ya que la máxima probabilidad de demanda para el día siguiente ronda 3 y 4 
    con poisson(D, 4), pero si ya tenemos mucho inventario, no es necesario comprar, ya que 
    primero se debe de vender lo que ya esta en almacen para no pagar tanto por almacenamiento

    6. ¿Como se comporta la función de valor de estado V(s)?
    Como un proceso iterativo que absorve la recompensa inmediata, pero mientras más iteraciones
    pasa, el valor comienza a recordar a través de las transiciones para capturar lo que gano hoy
    más lo que espero ganar mañana

    7. ¿Cómo cambiaría la política si la variabilidad de la demanda (lambda) aumenta de 4 a 8?
    Aumentan las acciónes, ya que es más probable que el día siguiente la demanda sea mayor
"""