"""
Para desarrollar el problema del inventario.

"""

from MDPs import MDP, iteracion_valor
from random import random, randint

class Inventario(MDP):
    """
    Clase que representa un MDP simple para un problema de inventario.

    Estado:
        s = unidades disponibles en inventario.

    Acciones:
        - 'no_ordenar': no se repone inventario.
        - 'ordenar': se repone una cantidad fija de unidades.

    Demanda:
        Se modela con una distribución discreta alrededor de lambda_.
        La transición depende de cuántas unidades se venden y de si se ordena.

    Recompensa:
        Incluye ingreso por ventas, costo por ordenar, costo por mantener inventario
        y penalización si no se alcanza a cubrir toda la demanda.
    """
    
    def __init__(self, gamma,lambda_, meta): 
        self.gamma = gamma
        self.lambda_ = lambda_
        self.capacidad = meta
        self.estados = tuple(range(0, self.capacidad + 1))
        self.cantidad_orden = max(1, self.capacidad // 2)
        self.precio_venta = 10
        self.costo_orden = 12
        self.costo_mantenimiento = 1
        self.penalizacion_faltante = 8

        # Demanda discreta aproximada alrededor de lambda_.
        self.demandas = tuple(range(0, min(self.capacidad, int(2 * self.lambda_)) + 1))
        if not self.demandas:
            self.demandas = (0,)
        self.prob_demandas = self._calcular_prob_demandas()

    def _calcular_prob_demandas(self):
        """Distribución discreta simple centrada en lambda_."""
        pesos = {d: 1 / (1 + abs(d - self.lambda_)) for d in self.demandas}
        total = sum(pesos.values())
        return {d: pesos[d] / total for d in self.demandas}

    def estado_inicial(self):
        return randint(0, self.capacidad)
        
    
    def acciones_legales(self, s):
        return ['no_ordenar', 'ordenar']
    
    def recompensa(self, s, a, s_):
        inventario_despues_de_ordenar = min(
            self.capacidad,
            s + (self.cantidad_orden if a == 'ordenar' else 0)
        )
        demanda_estimada = self.lambda_
        ventas_esperadas = min(inventario_despues_de_ordenar, demanda_estimada)
        faltante_esperado = max(0, demanda_estimada - inventario_despues_de_ordenar)

        ingreso = self.precio_venta * ventas_esperadas
        costo_orden = self.costo_orden if a == 'ordenar' else 0
        costo_mantenimiento = self.costo_mantenimiento * s_
        costo_faltante = self.penalizacion_faltante * faltante_esperado

        return ingreso - costo_orden - costo_mantenimiento - costo_faltante
        
    def prob_transicion(self, s, a, s_):
        inventario_despues_de_ordenar = min(
            self.capacidad,
            s + (self.cantidad_orden if a == 'ordenar' else 0)
        )

        prob = 0
        for demanda, p_demanda in self.prob_demandas.items():
            siguiente_estado = max(0, inventario_despues_de_ordenar - demanda)
            if siguiente_estado == s_:
                prob += p_demanda
        return prob
                
    def es_terminal(self, s):
        return False


if __name__ == "__main__":

    inventario = Inventario(gamma=0.9, lambda_=4, meta=20)

    pi_star, V = iteracion_valor(inventario, epsilon=0.01)

    print("-" * 60)
    print("Estado".center(20) + "Acción".center(20) + "Valor".center(20))
    print("-" * 60 )
    for s in pi_star:
        print(f"{s:^20}{pi_star[s]:^20}{V[s]:^20.2f}")
    print("-" * 60)


"""
Contesta las preguntas aquí mismo (haz espacio entre las preguntas):

1. ¿Cómo se comportan las transiciones y las ganancias para casos específicos de s y a?

Las transiciones dependen del inventario actual s, de la acción elegida y de la demanda.
Si la acción es no_ordenar, el siguiente estado tiende a ser menor que s, porque se venden
unidades y el inventario baja. Si la acción es ordenar, primero se agrega una cantidad fija
al inventario sin pasar la capacidad máxima, y después se resta la demanda.

La ganancia aumenta cuando se pueden vender unidades suficientes para cubrir la demanda,
pero disminuye si hay costos de ordenar, costos de mantener mucho inventario o penalizaciones
por faltantes. Por eso, ordenar no siempre es lo mejor: conviene principalmente cuando el
inventario actual es bajo.

2. ¿Qué pasa si hay mucho almacén?

Cuando hay mucho inventario, la política óptima normalmente evita ordenar. Esto tiene sentido
porque ya hay suficientes unidades para cubrir la demanda esperada. Además, tener demasiado
inventario genera costo de mantenimiento, por lo que acumular más producto puede reducir la
ganancia total.

3. ¿Qué pasa si hay muy poco o estamos sin almacén?

Cuando hay poco inventario o inventario cero, la política óptima tiende a ordenar. Si no se
ordena, es probable que no se pueda cubrir la demanda y aparezcan penalizaciones por faltantes.
En esos estados, el beneficio esperado de reponer inventario suele ser mayor que el costo de
hacer el pedido.

4. ¿Existe un punto donde la ganancia sea máxima?

Sí. La ganancia suele ser máxima en un nivel intermedio de inventario: suficiente para cubrir la
demanda esperada, pero no tan alto como para pagar demasiado costo de almacenamiento. Ese punto
representa un equilibrio entre vender lo más posible, evitar faltantes y no acumular inventario
innecesario.

---

5. ¿Cómo se ve la política óptima? ¿Tiene sentido?

La política óptima debe verse como una regla de umbral: ordenar cuando el inventario está por
debajo de cierto nivel y no ordenar cuando el inventario ya es suficiente. Sí tiene sentido,
porque en inventarios bajos el riesgo de faltantes es alto, mientras que en inventarios altos
ordenar solo aumenta costos.

6. ¿Cómo se comporta la función de valor de estado V(s)?

La función V(s) generalmente aumenta cuando se pasa de inventario muy bajo a inventario suficiente,
porque hay más posibilidad de cubrir la demanda y obtener ingresos. Después de cierto punto,
puede estabilizarse o incluso disminuir relativamente, porque demasiado inventario implica costos
de mantenimiento. Por eso V(s) no necesariamente crece para siempre; depende del balance entre
ventas, costos y penalizaciones.

7. ¿Cómo cambiaría la política si la variabilidad de la demanda, lambda, aumenta de 4 a 8?

Si lambda aumenta de 4 a 8, la demanda esperada es mayor. Entonces la política óptima tendería a
ordenar en más estados y a mantener niveles de inventario más altos. El umbral para ordenar se
movería hacia arriba, porque con mayor demanda hay más riesgo de quedarse sin inventario. También
podría aumentar la importancia de evitar faltantes, aunque mantener más inventario implique un
mayor costo de almacenamiento.
"""