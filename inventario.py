import math
from MDPs import MDP, iteracion_valor

class Inventario(MDP):
    """
    Clase que representa un MDP para el problema de inventario de Necroelectronica.
    """
    
    def __init__(self, gama=0.95, lambda_=4, cap_max=20, min_backlog=-10):
        self.gama = gama
        self.lambda_ = lambda_
        self.cap_max = cap_max
        self.min_backlog = min_backlog
        
        # Espacio de estados S = {-10, -9, ..., 20}
        self.estados = tuple(range(self.min_backlog, self.cap_max + 1))
        
        # Precalcular la distribución de Poisson para eficiencia (truncada en d=50)
        self.poisson = {}
        for d in range(50):
            prob = math.exp(-self.lambda_) * (self.lambda_ ** d) / math.factorial(d)
            self.poisson[d] = prob
            
    def acciones_legales(self, s):
        # La capacidad del estante es cap_max (20). 
        # Si s es negativo, podemos pedir hasta cap_max - s para llenar la bodega,
        # ya que al llegar el pedido, las unidades cubren el backlog primero.
        max_order = self.cap_max - s
        return list(range(0, max_order + 1))
    
    def prob_transicion(self, s, a, s_):
        # x es el inventario disponible justo en la mañana antes de la demanda
        x = s + a
        
        if s_ > x:
            return 0.0 # Es imposible que el inventario aumente sin pedir
            
        d = x - s_ # Demanda inferida
        
        if s_ > self.min_backlog:
            # Si no tocamos el fondo del backlog, la demanda fue exactamente d
            return self.poisson.get(d, 0.0)
        elif s_ == self.min_backlog:
            # Si llegamos a -10, significa que la demanda fue >= d
            prob_menor = sum(self.poisson.get(i, 0.0) for i in range(d))
            return 1.0 - prob_menor
            
        return 0.0
                
    def recompensa(self, s, a, s_):
        # Unidades físicamente transaccionadas = (Inventario inicial del día) - (Inventario final)
        unidades_vendidas = (s + a) - s_
        ingreso = 150 * unidades_vendidas
        
        # Costo de pedido: Fijo ($40) + Variable ($80 * a)
        costo_pedido = (40 + 80 * a) if a > 0 else 0
        
        # Costo de mantenimiento ($5 por unidad sobrante)
        costo_almacen = 5 * s_ if s_ > 0 else 0
        
        # Costo de backlogging ($15 por unidad faltante)
        costo_backlog = 15 * (-s_) if s_ < 0 else 0
        
        # La pérdida de oportunidad (margen de $70) está implícita:
        # Al no vender las unidades perdidas, no sumamos los $150 de ingreso,
        # pero tampoco pagamos los $80 de costo. El margen ya está restado del beneficio neto.
        
        return ingreso - costo_pedido - costo_almacen - costo_backlog

    def es_terminal(self, s):
        # El problema es de horizonte infinito, el negocio continúa día a día.
        return False

if __name__ == "__main__":
    inventario = Inventario(gama=0.95, lambda_=4)
    pi_star, V = iteracion_valor(inventario, epsilon=1e-4, max_iter=1000)

    print("-" * 60)
    print("Estado".center(20) + "Acción".center(20) + "Valor".center(20))
    print("-" * 60 )
    for s in pi_star:
        print(f"{s:^20}{pi_star[s]:^20}{V[s]:^20.2f}")
    print("-" * 60)


"""
Contesta las preguntas aquí mismo (has espacio entre las preguntas):

1. ¿Cómo se comporta las transiciones y las ganancias para casos específicos de $s$ y $a$? 
R: Si s y a suman un nivel alto, las transiciones se concentran en estados estables dictados por la media del parámetro lambda.
 Las ganancias son altas por ingresos de ventas estables, balanceadas por costos bajos de almacén. En niveles de s+a muy bajos,
   la transición decae hacia estados negativos (backlog), activando penalizaciones asimétricas severas.

2. ¿Qué psa si hay mucho almacen? 
R: Un exceso de inventario remanente (estados cercanos a 20) genera penalizaciones diarias acumulativas por mantenimiento ($5 por unidad).
 Esto mitiga el margen de beneficio neto, forzando al algoritmo a elegir la acción a=0 para vaciar el estante usando la demanda orgánica antes 
 de volver a incurrir en costos fijos.

3. ¿Que pasa si hay muy poco o estamos sin almacen? 
R: Al caer en desabasto o backlog, el negocio asume penalizaciones de logística ($15 por unidad faltante). Si se llega al límite inferior s=-10,
 cualquier demanda adicional satura el sistema y se pierde por completo la oportunidad del margen de ganancia de $70.
   El MDP mitiga esto ejecutando pedidos de gran escala para salir de la zona de penalización.

4. ¿Existe un punto donde la ganancia sea máxima? 
R: Sí, se localiza un punto de balance óptimo donde el nivel de inventario total amortiza el costo de 
almacenamiento y cubre eficientemente el riesgo de desabasto bajo la distribución estocástica.

---

5. ¿Cómo se ve la política óptima? ¿Tiene sentido?
R: Adopta la estructura de una política de umbral (s, S). Indica no pedir nada (a=0) mientras el stock se mantenga positivo para evadir el costo fijo de $40,
 pero si cae a 0 o menos, gatilla un pedido lo suficientemente grande para regresar a los niveles ideales de amortización de la demanda.
   Tiene absoluto sentido comercial.

6. ¿Como se comporta la función de valor de estado V(s)?
R: Es monótonamente creciente y cóncava. Su valor mínimo se encuentra en s=-10 producto de la acumulación de deudas de producto,
 ascendiendo de manera pronunciada conforme se transiciona hacia un inventario positivo, estabilizándose en los niveles más altos del almacén.

7. ¿Cómo cambiaría la política si la variabilidad de la demanda (lambda) aumenta de 4 a 8?
R: Al duplicarse el consumo de inventario diario, los umbrales de la política óptima se desplazarían hacia arriba.
 El sistema requerirá reordenar cantidades sustancialmente mayores con mucha más anticipación para blindar la operación contra las penalizaciones por desabasto.
"""