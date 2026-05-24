"""
Para desarrollar el problema del inventario.

"""

from MDPs import MDP, iteracion_valor

class Inventario(MDP):
    """  
    """
    def __init__(self, gama, lambda_):
        self.gama = gama
        self.lambda_ = lambda_
        self.k_max = 15
        self.estados = [i for i in range(-10, 21)]
    
    def acciones_legales(self, s):
        rango = 20 - s
        return [i for i in range(rango + 1)]
    
    def recompensa(self, s, a, s_):
        from math import factorial, exp

        def precio_venta(s, a):
            suma_esperada = 0.0
            
            disponible = max(s + a, 0)
            
            for k in range(self.k_max + 1):
                probabilidad = (exp(-self.lambda_) * (self.lambda_ ** k)) / factorial(k)
                
                vendido = min(disponible, k)
            
                suma_esperada += (150 * vendido) * probabilidad
                
            return suma_esperada
               
        def costo_compra_mas_pedido(a):
            costo = a * 80
            return costo + 40 if a > 0 else costo
        
        def costo_almacenamiento(s,a):
            suma_esperada = 0.0

            for k in range(self.k_max + 1):
                probabilidad = (exp(-self.lambda_) * (self.lambda_ ** k)) / factorial(k)
                suma_esperada += max((s + a)-k, 0) * probabilidad
                
            return 5 * suma_esperada
        
        def costo_inv_negativo(s,a):
            suma_esperada = 0.0

            for k in range(self.k_max + 1):
                probabilidad = (exp(-self.lambda_) * (self.lambda_ ** k)) / factorial(k)
                suma_esperada += max(k - (s + a), 0) * probabilidad

            return 85 * suma_esperada
        
        return precio_venta(s,a) + (-costo_compra_mas_pedido(a)) + (-costo_almacenamiento(s,a)) + (-costo_inv_negativo(s,a))
         
    def prob_transicion(self, s, a, s_):
        from math import factorial, exp
        
        D = s + a - s_

        if D >= 0 and s_ > -10:
            probabilidad = (exp(-self.lambda_) * (self.lambda_ ** D)) / factorial(D)
            return probabilidad
            
        if s_ > s + a:
            return 0.0
        
        if s_ == -10:
            suma_esperada = 0.0

            for k in range((s+a+10), self.k_max+1):
                probabilidad = (exp(-self.lambda_) * (self.lambda_ ** k)) / factorial(k)
                suma_esperada += probabilidad

            return suma_esperada

        return 0.0
             
    def es_terminal(self, s):
        return False # No hay un estado terminal

if __name__ == "__main__":

    inventario = Inventario(0.95, 4)

    pi_star, V = iteracion_valor(inventario)

    print("-" * 60)
    print("Estado".center(20) + "Acción".center(20) + "Valor".center(20))
    print("-" * 60 )
    for s in pi_star:
        print(f"{s:^20}{pi_star[s]:^20}{V[s]:^20.2f}")
    print("-" * 60)


"""
Contesta las preguntas aquí mismo (has espacio entre las preguntas):

1. ¿Cómo se comportan las transiciones y las ganancias para casos específicos de $s$ y $a$? 
2. ¿Qué pasa si hay mucho almacen? 
3. ¿Que pasa si hay muy poco o estamos sin almacen? 
4. ¿Existe un punto donde la ganancia sea máxima?  
---
5. ¿Cómo se ve la política óptima? ¿Tiene sentido?
6. ¿Como se comporta la función de valor de estado V(s)?
7. ¿Cómo cambiaría la política si la variabilidad de la demanda (lambda) aumenta de 4 a 8?

"""