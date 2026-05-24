"""
Para desarrollar el problema del inventario.

"""

from MDPs import MDP, iteracion_valor
import math

class Inventario(MDP):
    """
    Clase que representa un MDP para el problema de inventario.
    
    El gerente decide cuántas unidades pedir para minimizar costos de 
    almacenamiento y backlogging, maximizando la ganancia.
    
    """    
    
    def __init__(self, gamma, lambda_, precio_venta=150, costo_compra=80, 
                 costo_fijo_pedido=40, costo_almacen=5, costo_backlog=15, 
                 capacidad=20, backlog_max=10, k_max=20):
        """
        parámetros:
        gamma: factor de descuento (0.95)
        lambda_: media de la demanda poisson (4)
        precio_venta, costo_compra, costo_fijo_pedido, costo_almacen, costo_backlog
        capacidad: máxima unidades en almacén (20)
        backlog_max: límite inferior del inventario (negativo), ej. 10 -> -10
        k_max: truncamiento de la distribución poisson (probabilidad acumulada ~1)
        """         
        # generar espacio de estados: desde -backlog_max hasta capacidad
        estados = list(range(-backlog_max, capacidad + 1))

        # llamar al constructor de la clase base mdp
        super().__init__(estados, gamma)
        
        # guardar parámetros adicionales
        self.lambda_ = lambda_
        self.precio_venta = precio_venta
        self.costo_compra = costo_compra
        self.costo_fijo_pedido = costo_fijo_pedido
        self.costo_almacen = costo_almacen
        self.costo_backlog = costo_backlog
        self.capacidad = capacidad
        self.backlog_max = backlog_max
        self.k_max = k_max

        # precalcular probabilidades poisson hasta k_max
        self.poisson_probs = {}
        total = 0.0
        for k in range(k_max + 1):
            p = math.exp(-lambda_) * (lambda_ ** k) / math.factorial(k)
            self.poisson_probs[k] = p
            total += p
        # normalizar para que sume exactamente 1 (aunque la suma ya es muy cercana)
        for k in self.poisson_probs:
            self.poisson_probs[k] /= total
    
    def acciones_legales(self, s):
        """
        devuelve lista de acciones factibles dado el estado s.
        la acción es la cantidad a pedir (entero >= 0).
        restricción: s + a <= capacidad (no exceder almacén al recibir el pedido).
        """
        if s > self.capacidad:
            return [0]  # seguridad, no debería ocurrir
        max_a = self.capacidad - s
        return list(range(0, max_a + 1))

    def prob_transicion(self, s, a, s_):
        """
        probabilidad de transición de s a s_ con acción a.
        acumula la probabilidad de la cola en el estado límite inferior.
        """
        if s_ not in self.estados:
            return 0.0
            
        d = s + a - s_
        if d < 0:
            return 0.0 # no es posible que el inventario crezca sin pedir

        # si el estado de llegada no es el limite inferior (-10)
        if s_ > -self.backlog_max:
            return self.poisson_probs.get(d, 0.0)
            
        # si el estado de llegada es el límite inferior (-10)
        # acumulamos la probabilidad de que la demanda sea d o mayor, lo que llevaría a s' <= -10
        elif s_ == -self.backlog_max:
            # 1.0 menos la probabilidad de todas las demandas menores a d
            prob_menores = sum(self.poisson_probs.get(k, 0.0) for k in range(d))
            return max(0.0, 1.0 - prob_menores)

    def recompensa(self, s, a, s_):
        """
        recompensa inmediata por la transición (s, a, s_).
        se calcula con la demanda d = s + a - s_ (determinística dada la transición).
        incluye ingresos por ventas, costos de pedido, almacenamiento, backlog
        y pérdida de oportunidad (margen no ganado).
        """
        d = s + a - s_
        if d < 0:
            return -1e9  # transición imposible (castigo grande)

        I = s + a
        ventas = min(d, max(0, I))
        ingresos = self.precio_venta * ventas

        # costo de pedido
        if a > 0:
            costo_pedido = self.costo_fijo_pedido + self.costo_compra * a
        else:
            costo_pedido = 0

        # costo de almacenamiento (por inventario final positivo)
        costo_almacen = self.costo_almacen * max(s_, 0)

        # costo de backlog (por inventario final negativo)
        costo_backlog = self.costo_backlog * max(-s_, 0)

        # pérdida por demanda no satisfecha en el momento (margen no ganado)
        margen_unitario = self.precio_venta - self.costo_compra  # 70
        demanda_no_servida = d - ventas
        perdida_oportunidad = margen_unitario * demanda_no_servida

        recomp = ingresos - costo_pedido - costo_almacen - costo_backlog - perdida_oportunidad
        return recomp

    def es_terminal(self, s):
        """
        Como el negocio abre todos los días, seguira siendo False.
        """
        return False

if __name__ == "__main__":
    gamma = 0.95
    lambda_ = 4
    epsilon = 1e-4

    inventario = Inventario(gamma, lambda_)

    pi_star, V = iteracion_valor(inventario, epsilon=epsilon, max_iter=1000, debug=False)

    print("-" * 60)
    print("Estado".center(20) + "Acción".center(20) + "Valor".center(20))
    print("-" * 60 )
    for s in pi_star:
        print(f"{s:^20}{pi_star[s]:^20}{V[s]:^20.2f}")
    print("-" * 60)


"""
Contesta las preguntas aquí mismo (has espacio entre las preguntas):

1. ¿Cómo se comporta las transiciones y las ganancias para casos específicos de $s$ y $a$? 

   La transición depende de una demanda Poisson con media 4. si pides a unidades,
   al día siguiente antes de la demanda tienes s + a. luego llega la demanda y el nuevo 
   estado es s' = s + a - d, pero si se pasa de -10 se acumula ahí. la ganancia de esa 
   transición se calcula con los ingresos por ventas (150 por unidad vendida),
   menos el costo fijo de 40 si pediste algo, menos 80 por cada unidad pedida, menos 5 
   por cada unidad que sobra al final, menos 15 por cada unidad faltante (backlog)
   y menos 70 por cada unidad que no se pudo vender en el momento (margen perdido).
   Entonces, si la acción ajusta bien el inventario para que la demanda no deje mucho 
   sobrante ni mucho faltante, la ganancia es mayor.

2. ¿Qué pasa si hay mucho almacen? 

   si s es alto, digamos 10 o más, ya no conviene pedir porque la acción legal máxima 
   es poca o cero. En los resultados, para s >= 6 la política dice a = 0. Tiene sentido
   porque si ya tienes mucho inventario, pedir más solo te genera costo de almacenamiento 
   (5 por unidad) y el riesgo de que no se venda. Además, si pides aunque sea 1 unidad
   pagas el costo fijo de 40, que es caro. mejor dejarlo así y que la demanda poco a poco 
   baje el inventario.

3. ¿Que pasa si hay muy poco o estamos sin almacen? 

   Con s bajo o negativo (backlog) conviene pedir cantidades grandes: se aumenta
   s + a para cubrir la demanda esperada y reducir backlog (15 por unidad al
   cierre) y margen perdido. En los resultados, para s entre -10 y 5 la acción
   óptima cumple s + a = 9: se rellena hasta un nivel objetivo antes de la
   demanda del día siguiente. Esto es porque el castigo por faltantes es fuerte: 15 de backlog 
   más 70 de margen perdido, total 85 por cliente no atendido. Conviene pagar el costo fijo de 40 
   y los 80 por unidad para evitar esas penalizaciones.

4. ¿Existe un punto donde la ganancia sea máxima?  

   Cuando la demanda es igual al inventario, se venderia todo sin tener sobrantes
   ni faltantes, por lo que la ganancia inmediata es máxima. En terminos de la funcion V(s), el valor máximo
   se alcanza en s = 20, donde V(20) = 5574.14. Esto pasa porque no hay un costo por tener inventario inicial,
   solo se cobra almacenamiento al final del día, entonces empezar con más inventario ayuda a evitar penalizaciones futuras.

5. ¿Cómo se ve la política óptima? ¿Tiene sentido?

   Para s <= 5 se pide la cantidad que deja s + a = 9; para s >= 6, a = 0. 
   esta politica tiene sentido porque el costo fijo de pedido es de 40, lo que 
   desincentiva pedir poco, es mejor esperar a que baje el inventario para pedir y 
   hacer un pedido grande. El 9 es razonable considerando que la demanda media es 4
   por lo que cubre la demanda sin generar un exceso de almacenamiento.

6. ¿Como se comporta la función de valor de estado V(s)?

   La función V(s) es creciente en todo el dominio. Desde s = -10 hasta s = 5, el incremento
   entre estados consecutivos es exactamente 80. Esto se debe a que la política obliga a pedir
   una unidad adicional para llegar al nivel objetivo, y el costo de compra es 80. A partir de s = 6
   la política ya no pide nada, y el crecimiento se vuelve más suave, pero sigue siendo positivo.
   El valor máximo se alcanza en s = 20, donde V(20) = 5574.14.El comportamiento refleja que tener
   más inventario inicial reduce el riesgo de caer en backlog y pagar penalizaciones.

7. ¿Cómo cambiaría la política si la variabilidad de la demanda (lambda) aumenta de 4 a 8?

   Con lambda=4 la política pedía hasta llegar a 9 y dejaba de pedir desde s=6. con lambda=8
   el nivel objetivo subió a 13, es decir, para los estados desde -10 hasta 10 la acción es la que
   completa hasta 13 (por ejemplo, en s=-10 pide 23, en s=0 pide 13, en s=10 pide 3).El punto donde 
   ya no se pide también se corrió hacia arriba: ahora a=0 para s>=11. Los valores v(s) también son más altos, por ejemplo en s=20 
   antes era 5574 y ahora es 10804. esto tiene sentido porque la demanda promedio es mayor (8 en lugar de 4) y también hay más variabilidad
   entonces se necesita más inventario de seguridad para evitar el backlog y la pérdida de margen, que suman 85 por unidad no atendida. 
   Por eso la política se vuelve más agresiva y el nivel objetivo sube de 9 a 13.
   
"""