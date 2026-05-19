"""
Para desarrollar el problema del inventario.

"""
from MDPs import MDP, iteracion_valor

class Inventario(MDP):
    """
    Clase que representa un MDP para el problema del inventario.

    Objetivo:
    Se busca decidir cada tarde cuántas unidades pedir al proveedor para maximizar el beneficio a largo plazo

    """    
    
    def __init__(self, gamma=0.95, lambda_=4, precio_de_venta=150, costo_de_compra=80, costo_fijo_de_pedido=40,
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
        self.gama = gamma
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
        # TODO: Completar este método
        pass
        
    def prob_transicion(self, s, a, s_):
        # TODO: Completar este método
        pass
                
    def es_terminal(self, s):
        return False


if __name__ == "__main__":

    inventario = Inventario(0.95, 4, ...)  #TODO: Agregar lo que se requiera

    pi_star, V = iteracion_valor(inventario, ...) #TODO: Agregar lo que se requiera

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