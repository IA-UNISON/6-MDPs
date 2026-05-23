"""
Para desarrollar el problema del inventario.

"""
import math
from MDPs import MDP, iteracion_valor

class Inventario(MDP):
    """
    Clase que representa un MDP para el problema del inventario.

    Objetivo:
    Se busca decidir cada tarde cuántas unidades pedir al proveedor para maximizar el beneficio a largo plazo

    """    
    
    def __init__(self, gama=0.95, lambda_=4, precio_de_venta=150, costo_de_compra=80, costo_fijo_de_pedido=40,
                 costo_de_almacenamiento=5, costo_de_backlogging=15):
        """
        Inicializando las variables a utilizar:
        Gama: Es el factor de descuento
        Lambda_: Es la tasa promedio de la demanda. Representa la cantidad de clientes que llegan a comprar
        Precio de Venta: $150.00 por unidad vendida.
        Costo de Compra: $80.00 por unidad pedida al proveedor.
        Costo Fijo de Pedido: $40.00 por cada pedido realizado
        Costo de Almacenamiento: $5.00 por cada unidad que se quede en el estante al final del día.
        Costo de Backlogging (Inventario Negativo): Si la demanda supera las existencias, los clientes aceptan esperar,
        pero la empresa incurre en un costo de "buena voluntad" y logística de $15.00 por unidad faltante al final del día.
        """
        self.gama = gama
        self.lambda_ = lambda_
        self.estados = tuple(range(-10, 21))

        self.precio_de_venta = precio_de_venta
        self.costo_de_compra = costo_de_compra
        self.costo_fijo_de_pedido = costo_fijo_de_pedido
        self.costo_de_almacenamiento = costo_de_almacenamiento
        self.costo_de_backlogging = costo_de_backlogging
        self.perdida = self.precio_de_venta - self.costo_de_compra

    def acciones_legales(self, s):
        """
        La acción utilizada (a) debe cumplir que el inventario en el almacén disponible al siguiente dia no exceda
        la capacidad de 20
        """
        lim = 20 - s
        return list(range(0, lim + 1))
    
    def recompensa(self, s, a, s_):
        inventario_sig_dia = s + a
        demanda_real = inventario_sig_dia - s_
        existencias = max(0, inventario_sig_dia)

        costo_al_pedir = (self.costo_fijo_de_pedido + self.costo_de_compra * a) if a > 0 else 0

        unidades_vendidas = max(0, min(existencias, demanda_real))
        ingresos = self.precio_de_venta * unidades_vendidas

        costos_finales = (self.costo_de_almacenamiento * s_) if s_ > 0 else (
            self.costo_de_backlogging * abs(s_) if s_ < 0 else 0)

        ventas_a_deber = max(0, demanda_real - existencias)
        perdida_falta_stock = self.perdida * ventas_a_deber

        return ingresos - costo_al_pedir - costos_finales - perdida_falta_stock
        
    def prob_transicion(self, s, a, s_):
        demanda = (s + a) - s_
        if demanda < 0: return 0

        lam = self.lambda_

        if s_ == -10:
            return sum((math.exp(-lam) * (lam ** k)) / math.factorial(k) for k in range(demanda, 31))

        return (math.exp(-lam) * (lam ** demanda)) / math.factorial(demanda)
                
    def es_terminal(self, s):
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
El comportamiento se define por la tensión entre pedir mercancía y esperar a la demanda, por medio de una 
distribución de Poisson basada en la demanda promedio donde lambda equivale a 4, entonces, si el inventario 
al inicio del día está muy cercano a la demanda real, la probabilidad de terminar el día con solidez y con 
cercanía a 0 es muy alta, lo cual maximiza la ganancia al capturar el ingreso por ventas sin necesidad de 
castigos, en el caso contrario si el inventario disponible al inicio del día es muy bajo las transiciones 
harán que se recorra a un estado negativo haciendo que se desestabilicen las ganancias por los castigos aplicados 


2. ¿Qué pasa si hay mucho almacen? 
Cuando nos encontramos en estados muy altos la mejor acción será a = 0, es decir, no pedir nada porque así evitamos
el costo_fijo_de_pedido de $40 y los $80 que pueden variar. Aunque se tenga la seguridad de cubrir la demanda y no
incurrir en ventas perdidas, las ganancias netas diarias se verán afectadas por el costo_de_almacenamiento ($5 por
cada unidad no vendida), sin embargo, los ingresos serán ganancia casi neta porque estás vendiendo mercancía que ya 
había sido pagada en días anteriores, entonces el sistema simplemente deja que el inventario se vacie normal para 
evitar incurrir en el costo fijo de pedido ($40).


3. ¿Que pasa si hay muy poco o estamos sin almacen? 
Si estamos en s = 0 o en estado de backlog, la política exigirá realizar pedidos grandes. El modelo optará automáticamente 
por absorber el costo fijo de pedido ($40) y el costo variable de compra ($80 por unidad) con tal de salir inmediatamente 
del backlogging y asegurar stock para el día siguiente. El algoritmo toma esta decisión porque los castigos monetarios 
por no tener inventario son muy altos: El sistema registra una pérdida de $85 por cada cliente no atendido 
(la suma de los $15 de penalización por backlogging más los $70 del margen de ganancia perdido).

4. ¿Existe un punto donde la ganancia sea máxima? 
Sí,  la ganancia máxima se da cuando el inventario inicial del día s + a empata exactamente con la demanda real, 
vendiendo todo el stock sin generar costos adicionales de almacenamiento, ni por tener penalizaciones por artículos
faltantes.

---

5. ¿Cómo se ve la política óptima? ¿Tiene sentido?
Si, se ve de manera que si tenemos 3 unidades o más, no pedimos nada (a = 0). Si tenemos 2 unidades o menos, pide 
exactamente lo necesario para llegar a un nivel objetivo. Tiene sentido de manera que en el código tiene un 
costo_fijo_de_pedido de $40. Si pidiéramos de 1 por 1 todos los días, pagaríamos esos $40 diarios. La política óptima 
decide "aguantar" sin pedir mercancía hasta que el inventario baje a un punto crítico, y entonces hace un pedido grande 
de golpe para diluir ese costo fijo.

6. ¿Como se comporta la función de valor de estado V(s)?
La función V(s) es creciente, es decir, el valor a largo plazo siempre aumenta conforme el estado s es mayor.
Se comporta de la siguiente manera:
Estados bajos y negativos: Iniciando en el peor escenario de deuda V(-10), el crecimiento de la función es lineal, donde
aumenta $80 por estado, ya que este es el precio obligado a pagarle al proveedor para reponer cada unidad faltante.
Estados altos: El crecimiento cambia a crecer cada vez menos, aunque el valor sigue subiendo hasta llegar a su máximo 
en V(20) lo que equivale a tener dinero en forma de mercancía pagada, cada pieza adicional aporta menos ganancia debido 
a los costos de almacenamiento acumulados.
 

7. ¿Cómo cambiaría la política si la variabilidad de la demanda (lambda) aumenta de 4 a 8?
- Inventario: Como ahora llegan en promedio 8 clientes diarios en lugar de 4, las acciones crecerán, es decir,
se empezará a pedir un nivel mucho más grande de inventario para evitar vaciarse.
- Multa: Al hacer que lambda = 8, el riesgo de caer en la multa de $70 es muy alto + $15 de backlogging, entonces 
la política cambiará de manera que se preferirá tener inventarios mucho más altos para evitar caer en el backlogging


"""