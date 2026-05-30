"""
Para desarrollar el problema del inventario.

"""

from MDPs import MDP, iteracion_valor
from math import exp, factorial

class Inventario(MDP):
    """
    Clase que representa un MDP para el problema del camión mágico.
    
    Si caminas, avanzas 1 con coso 1
    Si usas el camion, con probabilidad rho avanzas el doble de donde estabas
    y con probabilidad 1-rho te quedas en el mismo lugar. Todo con costo 2.
    
    El objetivo es llegar a la meta en el menor costo posible
    
    """    
    
    def __init__(self, gama, lambda_, s_min=-10, s_max=20, capacidad=20): 
        self.gama = gama
        self.lambda_ = lambda_
        self.estados = tuple(range(s_min, s_max + 1))
        self.capacidad = capacidad
    
    def acciones_legales(self, s):
        return range(0, self.capacidad - s + 1)
    
    def recompensa(self, s, a, s_):
        D = s + a - s_
        return (150 * min(s + a, D)) - (80 * a) - (40 * (1 if a > 0 else 0)) - (5 * max(s_, 0) + 15 * max(-s_, 0)) 
        
    def prob_transicion(self, s, a, s_):
        D = s + a - s_

        if (D < 0):
            return 0
        else:
            return (exp(-self.lambda_) * (self.lambda_ ** D))/factorial(D)
                
    def es_terminal(self, s):
        return False


if __name__ == "__main__":

    inventario = Inventario(gama=0.95, lambda_=8) 

    pi_star, V = iteracion_valor(inventario, epsilon=1e-4) 

    print("-" * 60)
    print("Estado".center(20) + "Acción".center(20) + "Valor".center(20))
    print("-" * 60 )
    for s in pi_star:
        print(f"{s:^20}{pi_star[s]:^20}{V[s]:^20.2f}")
    print("-" * 60)


"""
Contesta las preguntas aquí mismo (has espacio entre las preguntas):

1. ¿Cómo se comporta las transiciones y las ganancias para casos específicos de $s$ y $a$? 
Las transiciones dependen de la demanda Si la demanda es negativa, la probabilidad es 0. Si la demanda es positiva, la probabilidad usa la funcion de Poisson.
Las ganancias en el estado inicial las ganancias aumentan 80 pesos cada paso, en el estado final las ganancias disminuyeron a aproximadamente 60 pesos.

2. ¿Qué pasa si hay mucho almacen? 
Se deja de comprar, porque ya existe el suficiente stock para cubrir la demanda.

3. ¿Que pasa si hay muy poco o estamos sin almacen? 
Se ordena hasta llenar inventario.

4. ¿Existe un punto donde la ganancia sea máxima?  
Entre el estado 5 y 6 (con gama=0.95 y lambda_=4) 

---
5. ¿Cómo se ve la política óptima? ¿Tiene sentido?
Si tiene 5 o menos unidades, pide hasta que tengas el maximo de unidades. Si hay 6 o mas no pedimos nada.

6. ¿Como se comporta la función de valor de estado V(s)?
Sube conforme S sube, hay mas ventas posibles y eso nos da más valor futuro.

7. ¿Cómo cambiaría la política si la variabilidad de la demanda (lambda) aumenta de 4 a 8?
La demanda es mayor y la politica pidem más unidades en los estados, por eso consigue más dinero y deja de pedir unidades hasta el estado 10
"""