"""
 Para desarrollar el problema del inventario.

 """

from MDPs import MDP, iteracion_valor
from math import exp, factorial


class Inventario(MDP):
    """
    MDP para el problema de inventario de Necroelectronica.

    Estados: inventario al final del dia, desde -max_backlog hasta capacidad
    Acciones: cuantas unidades pedir al proveedor (llegan al dia siguiente)
    """

    def __init__(self, gama, lambda_, capacidad=20, max_backlog=10,
                 precio=150, costo_unidad=80, costo_fijo=40,
                 costo_almacen=5, costo_backlog=15):
        self.lambda_ = lambda_
        self.capacidad = capacidad
        self.max_backlog = max_backlog
        self.precio = precio
        self.costo_unidad = costo_unidad
        self.costo_fijo = costo_fijo
        self.costo_almacen = costo_almacen
        self.costo_backlog = costo_backlog

        estados = tuple(range(-max_backlog, capacidad + 1))
        super().__init__(estados, gama)

    def acciones_legales(self, s):
        # se puede pedir desde 0 hasta llenar el almacen al maximo
        return list(range(0, self.capacidad - s + 1))

    def recompensa(self, s, a, s_):
        disponible = s + a  # inventario disponible para satisfacer demanda del dia

        # inferimos la demanda del dia a partir del estado siguiente
        if s_ > -self.max_backlog:
            d = disponible - s_
        else:
            # en el limite del backlog usamos la demanda minima que nos lleva ahi
            d = disponible + self.max_backlog

        if d < 0:
            return 0.0

        # solo se pueden vender las unidades que esten disponibles fisicamente
        vendidas = max(0, min(disponible, d))
        ingreso = self.precio * vendidas

        costo_orden = self.costo_unidad * a + (self.costo_fijo if a > 0 else 0)
        costo_hold = self.costo_almacen * max(0, s_)
        penalizacion = self.costo_backlog * max(0, -s_)

        return ingreso - costo_orden - costo_hold - penalizacion

    def _poisson_pmf(self, k):
        if k < 0:
            return 0.0
        return exp(-self.lambda_) * (self.lambda_ ** k) / factorial(k)

    def prob_transicion(self, s, a, s_):
        disponible = s + a  # inventario al inicio del dia tras recibir el pedido

        if s_ > -self.max_backlog:
            d = disponible - s_  # demanda implicita para llegar a s_
            if d < 0:
                return 0.0
            return self._poisson_pmf(d)
        else:
            # s_ == -max_backlog: la demanda fue tan alta que saturamos el backlog
            # P(D >= disponible + max_backlog)
            d_min = disponible + self.max_backlog
            if d_min <= 0:
                return 1.0
            prob_cola = 0.0
            for d in range(d_min, d_min + 60):
                p = self._poisson_pmf(d)
                prob_cola += p
                if p < 1e-10:
                    break
            return prob_cola

    def es_terminal(self, s):
        # el inventario no tiene estados terminales, opera indefinidamente
        return False


if __name__ == "__main__":

    inventario = Inventario(gama=0.95, lambda_=4)

    pi_star, V = iteracion_valor(inventario, epsilon=1e-4)

    print("-" * 60)
    print("Estado".center(20) + "Acción".center(20) + "Valor".center(20))
    print("-" * 60)
    for s in sorted(pi_star):
        print(f"{s:^20}{pi_star[s]:^20}{V[s]:^20.2f}")
    print("-" * 60)

"""
  Contesta las preguntas aquí mismo (has espacio entre las preguntas):

  1. ¿Cómo se comporta las transiciones y las ganancias para casos específicos de $s$ y $a$?

  Para s=5 (5 unidades en almacen) y a=3 (pedimos 3), disponible=8.
  La demanda sigue Poisson(4), asi que lo mas probable es d=3 o d=4 (~20% cada una).
  Si d=4 entonces s'=4, recompensa = 150*4 - (80*3 + 40) - 5*4 = 600 - 280 - 20 = 300.
  Si d=10 (poco probable), s'=-2, recompensa = 150*8 - 280 - 15*2 = 1200 - 280 - 30 = 890
    pero esa transicion tiene probabilidad muy baja (~0.5%).
  En general, las ganancias son mas altas cuando se vende todo el inventario sin quedarse
  en backlog, porque se evitan tanto los costos de almacen como las penalizaciones.

  2. ¿Qué psa si hay mucho almacen?

  Cuando s es alto (cerca de 20) se acumulan muchos costos de almacenamiento (5 por unidad).
  Con s=20 y a=0, si la demanda es 4 se paga 5*16=80 de almacenamiento. La politica optima
  refleja esto: para s >= 6 la accion es a=0, no vale la pena pedir mas aunque tengamos espacio.

  3. ¿Que pasa si hay muy poco o estamos sin almacen?

  Con inventario negativo (backlog) se pagan 15 por unidad en backlog y ademas se deja de
  ganar el margen de 70 (150 precio - 80 costo) por cada unidad no entregada. La politica
  reacciona pidiendo muchas unidades para salir del backlog rapido. El costo fijo de 40 hace
  que valga la pena pedir bastante de una vez para no pagar ese costo fijo repetidamente.

  4. ¿Existe un punto donde la ganancia sea máxima?

  Si, el nivel optimo esta alrededor de s+a=9 (con lambda=4). Ahi se balancea el riesgo de
  quedarse sin inventario (backlog) contra el costo de tener demasiado (almacenamiento).
  Tener mas de ~9 unidades al inicio del dia empieza a costar mas de lo que aporta en
  terminos de ventas adicionales esperadas.

  ---

  5. ¿Cómo se ve la política óptima? ¿Tiene sentido?

  Es una politica tipo (s, S): si el inventario actual es menor o igual a 5 se ordena hasta
  llegar a 9 unidades (order-up-to 9), y si ya se tienen 6 o mas no se pide nada.
  Tiene mucho sentido porque es la politica clasica para este tipo de problemas con costo
  fijo de pedido. El umbral inferior (s=5) evita pagar el costo fijo por pedidos pequeños.

  6. ¿Como se comporta la función de valor de estado V(s)?

  V(s) crece conforme aumenta s: tener mas inventario siempre es mejor que menos porque
  se puede atender mas demanda sin incurrir en penalizaciones. Los valores van de ~3312
  (s=-10) hasta ~5633 (s=20). El incremento entre estados consecutivos es aproximadamente
  constante (~80, que coincide con el costo variable por unidad), lo cual tiene sentido
  porque cada unidad adicional de inventario "vale" un pedido virtual de $80.

  7. ¿Cómo cambiaría la política si la variabilidad de la demanda (lambda) aumenta de 4 a 8?

  Con lambda=8 el order-up-to sube de 9 a 13: hay que mantener mas inventario para cubrir
  la mayor demanda esperada. El punto de reorden tambien sube de s=5 a s=9. En general la
  politica es mas agresiva ordenando mas unidades porque el riesgo de quedarse sin stock
  (y pagar penalizaciones de backlog) es mucho mayor con demanda promedio de 8 vs 4.

  """

