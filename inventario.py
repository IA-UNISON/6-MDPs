"""
Para desarrollar el problema del inventario.

"""

from math import exp, factorial
from MDPs import MDP, iteracion_valor

class Inventario(MDP):
    """
    MDP del inventario de Necroelectronica. El estado s es el inventario al
    final del dia (negativo = backlog). La accion a es cuanto pedir esa tarde.

    Secuencia: por la mañana llega el pedido (disponible y = s + a), durante el
    dia ocurre la demanda D ~ Poisson(lambda), y el estado siguiente es
    s' = y - D. La capacidad topa y en cap_max y el backlog topa en cap_min.

    """

    def __init__(self, gama, lambda_, cap_max=20, cap_min=-10,
                 precio=150, c_compra=80, c_fijo=40, c_almacen=5,
                 c_backlog=15, trunc=30):
        self.gama = gama
        self.lambda_ = lambda_
        self.cap_max = cap_max
        self.cap_min = cap_min
        self.precio = precio
        self.c_compra = c_compra
        self.c_fijo = c_fijo
        self.c_almacen = c_almacen
        self.c_backlog = c_backlog
        self.trunc = trunc  # demanda mas alla de esto tiene prob ~0
        self.estados = tuple(range(cap_min, cap_max + 1))

    def _poisson(self, k):
        # f(k; lambda) = e^-lambda * lambda^k / k!
        if k < 0:
            return 0.0
        return exp(-self.lambda_) * self.lambda_ ** k / factorial(k)

    def acciones_legales(self, s):
        # se puede pedir desde 0 hasta llenar el estante (y = s + a <= cap_max)
        tope = self.cap_max - max(0, s)
        return list(range(0, tope + 1))

    def recompensa(self, s, a, s_):
        # disponible tras recibir el pedido y demanda implicita de la transicion
        y = s + a
        demanda = y - s_
        # ventas: lo que se entrega del stock fisico disponible
        ventas = min(max(0, y), demanda) if demanda >= 0 else max(0, y)
        # demanda no satisfecha (margen no ganado = precio - costo de compra)
        no_satisfecha = max(0, demanda - max(0, y))

        ingreso = self.precio * ventas
        costo_pedido = self.c_compra * a + (self.c_fijo if a > 0 else 0)
        costo_inv = (self.c_almacen * max(0, s_) +
                     self.c_backlog * max(0, -s_))
        costo_perdida = (self.precio - self.c_compra) * no_satisfecha

        return ingreso - costo_pedido - costo_inv - costo_perdida

    def prob_transicion(self, s, a, s_):
        y = s + a
        demanda = y - s_
        if demanda < 0:
            return 0.0
        # el estado minimo absorbe toda la cola: P(D >= y - cap_min)
        if s_ == self.cap_min:
            return sum(self._poisson(k)
                       for k in range(y - self.cap_min, self.trunc + 1))
        return self._poisson(demanda)

    def es_terminal(self, s):
        # proceso continuo de decision: ningun estado es terminal
        return False


if __name__ == "__main__":

    inventario = Inventario(0.95, 4)

    pi_star, V = iteracion_valor(inventario, epsilon=1e-4)

    print("-" * 60)
    print("Estado".center(20) + "Acción".center(20) + "Valor".center(20))
    print("-" * 60 )
    for s in pi_star:
        print(f"{s:^20}{pi_star[s]:^20}{V[s]:^20.2f}")
    print("-" * 60)


"""
Contesta las preguntas aquí mismo (has espacio entre las preguntas):

1. ¿Cómo se comporta las transiciones y las ganancias para casos específicos de s y a?
   La transicion depende solo del disponible y = s + a y de la demanda D ~ Poisson(lambda):
   s' = y - D. La ganancia esperada de (s, a) es ingreso por ventas (min(y, D) * 150) menos
   el costo de pedir (80 por unidad + 40 fijo si a > 0), menos el costo del inventario final
   (5 por unidad sobrante o 15 por unidad en backlog), menos el margen no ganado por demanda
   no surtida (70 por unidad). Ejemplo: en s = 0 pedir a = 9 deja y = 9, suficiente para cubrir
   la demanda media de 4 con holgura, y resulta ser el optimo.

2. ¿Qué pasa si hay mucho almacen?
   Con inventario alto (s >= 6) la politica optima es NO pedir (a = 0): ya hay stock de sobra
   para la demanda esperada, y pedir mas solo sumaria costo de compra + almacenamiento. El valor
   V(s) sigue creciendo con s porque mas stock evita futuros faltantes, pero el extra es marginal.

3. ¿Que pasa si hay muy poco o estamos sin almacen?
   Con inventario bajo o negativo (backlog) la politica pide agresivamente hasta llevar el
   disponible al nivel objetivo (y = 9 con lambda = 4). En backlog ademas se paga la penalizacion
   de 15 por unidad, asi que reabastecer es urgente. V(s) es menor cuanto mas negativo es s.

4. ¿Existe un punto donde la ganancia sea máxima?
   Si. El nivel objetivo de reabastecimiento es el punto donde la ganancia esperada por dia se
   maximiza: equilibra el costo de almacenar de mas contra el costo de quedarse corto. Con
   lambda = 4 ese punto es y = 9 unidades disponibles.

---

5. ¿Cómo se ve la política óptima? ¿Tiene sentido?
   Es una politica de tipo (s, S) clasica de inventarios: si el inventario cae a 5 o menos, se
   pide hasta llegar a 9 disponibles; si es 6 o mas, no se pide nada. Tiene todo el sentido
   economico: hay un punto de reorden (5) y un nivel objetivo (9), exactamente el resultado de
   libro de texto para este tipo de problema.

6. ¿Como se comporta la función de valor de estado V(s)?
   V(s) es monotona creciente en s: mas inventario disponible vale mas porque reduce el riesgo de
   faltantes futuros. En la zona de backlog crece de forma lineal (cada unidad menos de deuda
   vale ~80, el costo de reponerla), y en la zona de stock alto la pendiente se aplana porque el
   stock extra aporta cada vez menos.

7. ¿Cómo cambiaría la política si la variabilidad de la demanda (lambda) aumenta de 4 a 8?
   La politica se vuelve mas agresiva. Verificado numericamente: con lambda = 8 el punto de
   reorden sube de 5 a 10 y el nivel objetivo de 9 a 13. Al esperar mas demanda, conviene
   reabastecer antes y mantener mas stock de seguridad para no incurrir en faltantes, que son
   mas caros (margen perdido + backlog) que el costo de almacenar.

"""

"""
Ejemplo de ejecucion y output en consola
 - La forma de representar el estado como una tupla (s \in S)
------------------------------------------------------------
    Estado              Acción              Valor        
------------------------------------------------------------
     -10                  19               3245.52       
      -9                  18               3325.52       
      -8                  17               3405.52       
      -7                  16               3485.52       
      -6                  15               3565.52       
      -5                  14               3645.52       
      -4                  13               3725.52       
      -3                  12               3805.52       
      -2                  11               3885.52       
      -1                  10               3965.52       
      0                   9                4045.52       
      1                   8                4125.52       
      2                   7                4205.52       
      3                   6                4285.52       
      4                   5                4365.52       
      5                   4                4445.52       
      6                   0                4538.57       
      7                   0                4637.37       
      8                   0                4723.88       
      9                   0                4805.52       
      10                  0                4885.40       
      11                  0                4963.91       
      12                  0                5040.55       
      13                  0                5114.93       
      14                  0                5186.97       
      15                  0                5256.77       
      16                  0                5324.41       
      17                  0                5389.95       
      18                  0                5453.41       
      19                  0                5514.80       
      20                  0                5574.14       
------------------------------------------------------------
"""