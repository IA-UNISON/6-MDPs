"""
Para desarrollar el problema del inventario.

"""

import math

from MDPs import MDP, iteracion_valor

class Inventario(MDP):
    """
    Clase que representa un MDP para el problema del camión mágico.
    
    Si caminas, avanzas 1 con coso 1
    Si usas el camion, con probabilidad rho avanzas el doble de donde estabas
    y con probabilidad 1-rho te quedas en el mismo lugar. Todo con costo 2.
    
    El objetivo es llegar a la meta en el menor costo posible
    
    """    
    
    def __init__(self, gama,lambda_,capacidad, costo_orden, costo_almacenamiento):
        self.gama = gama
        self.lambda_ = lambda_
        self.capacidad = capacidad
        self.costo_orden = costo_orden
        self.costo_almacenamiento = costo_almacenamiento
        self.estados = list(range(capacidad + 1))  
    
    def acciones_legales(self, s):
        return list(range(0, self.capacidad - s + 1))
    
    def recompensa(self, s, a, s_):
        return - (self.costo_orden * a + self.costo_almacenamiento * s_)
    
    def prob_poisson(self, k):
        if k < 0:
            return 0.0
        return (math.exp(-self.lambda_) * self.lambda_**k) / math.factorial(k)
        
    def prob_transicion(self, s, a, s_):
        inventario = s + a
        
        if s_ > 0:
            demanda = inventario - s_
            return self.prob_poisson(demanda)
        else:
            prob = 0.0
            for demanda in range(inventario + 1):
                prob += self.prob_poisson(demanda)
            return prob
                
    def es_terminal(self, s):
        return False


if __name__ == "__main__":
    inventario = Inventario(
        gama = 0.9,
        lambda_ = 2.0,
        capacidad = 20,
        costo_orden = 2,
        costo_almacenamiento = 1
    )

    pi_star, V = iteracion_valor(inventario, epsilon = 0.001)

    print("-" * 60)
    print("Estado".center(20) + "Acción".center(20) + "Valor".center(20))
    print("-" * 60 )
    for s in pi_star:
        print(f"{s:^20}{pi_star[s]:^20}{V[s]:^20.2f}")
    print("-" * 60)


"""
Contesta las preguntas aquí mismo (has espacio entre las preguntas):

1. ¿Cómo se comporta las transiciones y las ganancias para casos específicos de $s$ y $a$? 

Como las ventas siguen una distribución de Poisson, existen dos casos:
 - Si sobran productos, es decir s' > 0, significa que se vendieron exactamente la diferencia
    entre el inventario y el estado siguiente.
 - Si no sobran productos, es decir s' = 0, significa que la demanda fue tan alta que se vendieron
   todos los productos de la bodega.

   
2. ¿Qué pasa si hay mucho almacen? 

Si la bodega esta llena, va a salir caro porque el costo del almacenamiento se come tus ganancias, ya
que estas pagando espacio por productos que no se estan vendiendo rapidamente. Ademas, significa que
no vas a poder pedir mas productos. Y por ello el valor del estado empieza a caer cuando el inventario
se acerca a la capacidad maxima.


3. ¿Que pasa si hay muy poco o estamos sin almacen? 

Si no hay productos en el inventario y no hay pedidos, toda la demanda se pierde y te quedas en cero. La recompensa es cero,
pero el sistema queda en un estado vulnerable para el siguiente periodo. Si pides mucho de golpe, el costo
de hacer el pedido es alto, pero al menos dejas el negocio listo para vender y recuperarte luego.


4. ¿Existe un punto donde la ganancia sea máxima?  

Si, donde se logra un balance donde no pides tan seguido pero tampoco acumulas mucho.
Este punto se observa en la política óptima y en la función de valor de estado, donde se observa un aumento 
en el valor a medida que el inventario se acerca a este punto, y luego una disminución a medida que el inventario 
se acerca a la capacidad máxima.


---


5. ¿Cómo se ve la política óptima? ¿Tiene sentido?

Si tiene sentido, parece una escalera donde si tu inventario es bajo, pides mas productos y conforme el inventario 
aumenta, cada vez pides menos. Ya pasando cierto nivel no pides nada, para vender lo que tienes almacenado.


6. ¿Como se comporta la función de valor de estado V(s)?

El valor de estado empieza bajo cuando estas en cero. Mientras tienes un inventario moderado, el valor sube y se
pone en su punto maximo, ya que tienes productos para vender sin pagar de exceso por almacenarlos.


7. ¿Cómo cambiaría la política si la variabilidad de la demanda (lambda) aumenta de 4 a 8?

Si lambda pasa de 4 a 8, significa que el promedio de clientes que entran a comprar se duplico. Al tener tanta 
demanda estas en peligro de quedarte sin inventario. Entonces la politica se vuelve mas agresiva, y el algoritmo
va a recomendar pedir mas productos.


"""