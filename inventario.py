"""
Para desarrollar el problema del inventario.

"""
import math
from MDPs import MDP, iteracion_valor

class Inventario(MDP):
    """
    En este problema se modela la gestión de inventarios de una tienda. El objetivo es determinar la cantidad óptima de unidades a 
    pedir cada día para maximizar las ganancias a largo plazo, considerando los costos de compra, almacenamiento y backlog, así como las ganancias por ventas.   
    

    Estados:
        Representan la cantidad de unidades en el invnetario disponible, incluyendo backlog.

    Acciones:
        Cantidad de unidades que se pueden pedir.

    Recompensas:
        Se calculan considerando:
        - Ganancias por ventas
        - Costos de compra
        - Costos de almacenamiento
        - Costos de backlog

    Transiciones:
        La transición de estados depende de la demanda diaria, que se modela como una variable aleatoria con distribución de Poisson. 
        La cantidad de unidades vendidas se determina por la demanda y el inventario disponible, y el siguiente estado se calcula restando la demanda al inventario actual más las unidades compradas.
    """    
    
    def __init__(self, gama,lambda_, capacidad=20, inventario_min=-10,precio_venta=150,
                 costo_compra=80, costo_fijo=40, costo_almacen=5,costo_backlog=15): 
        self.gamma = gama
        self.lambda_ = lambda_
        self.capacidad = capacidad
        self.inventario_min = inventario_min
        self.precio_venta=precio_venta
        self.costo_compra=costo_compra
        self.costo_fijo=costo_fijo
        self.costo_almacen= costo_almacen
        self.costo_backlog= costo_backlog
        self.estados = tuple(range(self.inventario_min,self.capacidad+1))
        
    
    def acciones_legales(self, s):
        return range(max(0, self.capacidad - s + 1))
    
    def recompensa(self, s, a, s_):
        demanda = (s+a)-s_
        venta = min(demanda, max(s+a, 0))
        ganancia = venta * self.precio_venta 
        costo = a*self.costo_compra
        if a > 0:
            costo += self.costo_fijo
        if s_ > 0:
            costo += s_ * self.costo_almacen
        if s_ < 0:
            costo += abs(s_) * self.costo_backlog
        return ganancia - costo
        
    def prob_transicion(self, s, a, s_):
        demanda = (s+a)-s_
        if demanda < 0:
            return 0

        if s_ > self.inventario_min:

            return (self.lambda_ ** demanda* math.exp(-self.lambda_)) / math.factorial(demanda)

        prob = 0

        for d in range(demanda, demanda + 20):
            prob += (self.lambda_ ** d* math.exp(-self.lambda_)) / math.factorial(d)

        return prob 


    def es_terminal(self, s):
        return False 


if __name__ == "__main__":

    inventario = Inventario(gama=0.95, lambda_=4) 

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

Las transiciones y las ganancias dependen del estado actual del inventario (s) y la cantidad de unidades que se decida comprar (a). 
El siguiente estado se obtiene restando la demanda (que es la cantidad de unidades vendidas) al inventario actual más las unidades compradas.
Cuando el inventario es negativo el modelo suele elegir acciones grandes para evitar el costo de backlog, en cambio cuando el inventario 
es alto, se selecciona acciones cercanas a 0 para evitar el costos de almacenamiento. Las ganancias se obtienen de las ventas menos los costos 
de compra, almacenamiento y backlog. Permitiendo que el modelo encuentre una política que maximice las ganancias a largo plazo.


2. ¿Qué pasa si hay mucho almacen? 

Si hay mucho inventario almacenado aumentan los costos de almacenamiento, lo que puede llevar a una reducción en las ganancias. 
Por lo que el modelo prefiere reducir o detener los pedidos hasta que el inventario vuelva a bajar. El modelo calcula cuál acción genera 
la mayor recompensa considerando las ventas, los costos de compra, almacenamiento y backlog. 


3. ¿Que pasa si hay muy poco o estamos sin almacen? 

Si hay muy poco inventario aumenta el riesgo de no poder satisfacer la demanda de los clientes, lo que genera costos de backlog y reduce
las ganancias. Por ello el modelo aprende que conviene realizar pedidos más grandes para recuperar inventario y evitar pérdidas por faltantes. 
Pero cuando el inventario es muy bajo el modelo da prioridad a acciones que permitan satisfacer la demanda futura y reducir la probabilidad 
de quedarse sin productos disponibles. De esta manera busca mantener un equilibrio entre evitar faltantes y no generar costos excesivos 
por almacenamiento.


4. ¿Existe un punto donde la ganancia sea máxima?  

Si, cuando el inventario es cercana al inventario disponible, se pueden satisfacer las ventas sin generar demasiados sobrantes ni faltantes, 
por lo que la ganancia inmediata tiende a ser mayor. En la función el valor máximo se alcanza en estados con inventario alto donde el 
modelo tiene suficientes productos para cubrir demandas futuras y evitar penalizaciones por backlog. Esto ocurre porque el modelo busca 
equilibrar las ganancias por ventas con los costos de almacenamiento y backlog para maximizar el beneficio esperado a largo plazo.


5. ¿Cómo se ve la política óptima? ¿Tiene sentido?

Si tiene sentido, ya que cuando el inventario es negativo el modelo recomienda realizar pedidos grandes. Por ejemplo en el estado s=-10 la 
accion óptima es a=19 unidades, mientras que en s=0 recomienda pedir 9. Conforme el inventario aumenta, la cantidad pedida dismiuye hasta 
llegar a 0 a partir de s=6. Además los valores de V(s) aumentan conforme el inventario es mayor, indicando que tener suficiente inventario
genera una mayor recompensa.



6. ¿Como se comporta la función de valor de estado V(s)?

La función de valor V(s) aumenta conforme el inventario disponible es mayor. Donde los estados con inventario negativo tienen valores a 
menores, mientras que los estados con inventario alto tienen valores mas grandes. Esto ocurre porque tener mas inventario reduce la 
probabilidad de faltantes y costos de backlog, permitiendo satisfacer mejor la demanda futura. Además a partir de ciertos estados el modelo 
deja de realizar pedidos, ya que considera que el inventario actual es suficiente para maximizar la recompensa esperado a largo plazo.

7. ¿Cómo cambiaría la política si la variabilidad de la demanda (lambda) aumenta de 4 a 8?

Al aumentar lambda de 4 a 8 el modelo comienza a realizar pedidos más grandes y mantener niveles de inventario más altos para satisfacer 
la mayor demanda esperada y evitar costos de backlog.

"""