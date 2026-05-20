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

    inventario = Inventario(0.95, 4,)

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
2. ¿Qué psa si hay mucho almacen? 
3. ¿Que pasa si hay muy poco o estamos sin almacen? 
4. ¿Existe un punto donde la ganancia sea máxima?  
---
5. ¿Cómo se ve la política óptima? ¿Tiene sentido?
6. ¿Como se comporta la función de valor de estado V(s)?
7. ¿Cómo cambiaría la política si la variabilidad de la demanda (lambda) aumenta de 4 a 8?

"""