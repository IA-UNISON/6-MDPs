"""
Para desarrollar el problema del inventario.

"""

from MDPs import MDP, iteracion_valor
import math

def poisson_pmf(k, lam):
    if k < 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

class Inventario(MDP):
    """
    Clase que representa un MDP para el problema del camión mágico.
    
    Si caminas, avanzas 1 con coso 1
    Si usas el camion, con probabilidad rho avanzas el doble de donde estabas
    y con probabilidad 1-rho te quedas en el mismo lugar. Todo con costo 2.
    
    El objetivo es llegar a la meta en el menor costo posible
    
    """    
    
    def __init__(
         self,
        gama         = 0.95,   
        lambda_      = 4,      
        s_min        = -10,    
        s_max        = 20,     
        precio_venta = 150,    
        costo_unit   = 80,     
        costo_pedido = 40,    
        costo_hold   = 5,      
        costo_back   = 15,     
    ):
        estados = list(range(s_min, s_max + 1))
        super().__init__(estados, gama)  
 
        self.lambda_      = lambda_
        self.s_min        = s_min
        self.s_max        = s_max
        self.precio_venta = precio_venta
        self.costo_unit   = costo_unit
        self.costo_pedido = costo_pedido
        self.costo_hold   = costo_hold
        self.costo_back   = costo_back

        k, acum, probs = 0, 0.0, []
        while acum < 0.9999:
            p = poisson_pmf(k, lambda_)
            probs.append(p)
            acum += p
            k += 1
    
        total = sum(probs)
        self._prob_d = [p / total for p in probs]  

    def acciones_legales(self, s):
        a_max = max(self.s_max - s, 0)
        return list(range(0, a_max + 1))
    
    def recompensa(self, s, a, s_):
        x = s + a   # inventario disponible al inicio del día

        if x >= 0:
            vendidas = x - max(s_, 0)
        else:
            vendidas = 0
 
        ingreso  =  self.precio_venta * vendidas
        c_compra = -self.costo_unit   * a
        c_fijo   = -self.costo_pedido * (1 if a > 0 else 0)
        c_hold   = -self.costo_hold   * max(s_, 0)
        c_back   = -self.costo_back   * max(-s_, 0)
 
        return ingreso + c_compra + c_fijo + c_hold + c_back
        
    def prob_transicion(self, s, a, s_):
        x = s + a
        prob = 0.0
        for d, p in enumerate(self._prob_d):
            s_next = max(self.s_min, min(self.s_max, x - d))
            if s_next == s_:
                prob += p
        return prob
                
    def es_terminal(self, s):
        return False


if __name__ == "__main__":

    inventario = Inventario(
        gama=0.95,
        lambda_=4,
        s_min=-10,
        s_max=20,
        precio_venta=150,
        costo_unit=80,
        costo_pedido=40,
        costo_hold=5,
        costo_back=15,
    )

    pi_star, V = iteracion_valor(
        inventario,
        epsilon=1e-4,
        max_iter=1000,
        debug=True
    )

    print("-" * 60)
    print("Estado".center(20) +
          "Acción".center(20) +
          "Valor".center(20))
    print("-" * 60)

    for s in inventario.estados:
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