"""
Para desarrollar el problema del inventario.

"""
import math
from MDPs import MDP, iteracion_valor

def poisson_pmf(k, lambda_):
    if k < 0:
        return 0.0
    return math.exp(-lambda_)*(lambda_**k)/math.factorial(k)

class Inventario(MDP):
    """
    MDP para el problema de inventario

    Estado s: inventario neto al final del dia
    s > 0 -> unidades en almacen
    s = 0 -> sin stock
    s < 0 -> backlog

    Accion a: cuantas unidades pedir esa tarde
    """    
    
    def __init__(self, gama,lambda_, capacidad=20, backlog_max=10,
                 precio=150, costo_var=80, costo_fijo=40, costo_hold=5,
                 costo_back=15, margen_perd=70): 
        #Espacio de estados
        estados = list(range(-backlog_max, capacidad + 1))
        super().__init__(estados, gama)

        self.lambda_ = lambda_
        self.capacidad = capacidad
        self.backlog_max = backlog_max
        self.precio = precio
        self.costo_var = costo_var
        self.costo_fijo = costo_fijo
        self.costo_hold = costo_hold
        self.costo_back = costo_back
        self.margen_perd = margen_perd
        
        self.s_min = -backlog_max
        self.s_max = capacidad

        #Truncar la demanda donde la probalbilidad acumulada es -1
        self.D_max = 0
        while poisson_pmf(self.D_max, lambda_) > 1e-9:
            self.D_max += 1
    
    def acciones_legales(self, s):
        """
        A(s) = {0, 1, ..., capacidad - s}

        No puede pedir negativo y el almacen no puede recibir mas
        de lo que cabe
        """
        return list(range(0, self.capacidad - s + 1))
    
    def _ganancia(self, s, a, D):
        """
        Ganancia neta cuando en el estado s se pide a y la demanda es D
        """
        I = s + a #Inventario al inicio del dia
        sp = I -D #Inventario al final del dia

        ventas = self.precio * max(0, min(D, I))
        compra = self.costo_var * a
        fijo = self.costo_fijo * (1 if a > 0 else 0)
        hold = self.costo_hold * max(0, sp)
        back = self.costo_back * max(0, -sp)
        opp = self.margen_perd * max(0, D - max(0,I))

        return ventas - compra - fijo - hold - back - opp
    
    def prob_transicion(self, s, a, s_):
        D = (s+a) - s_

        if s_ == self.s_min:
            return sum(
                poisson_pmf(d, self.lambda_)
                for d in range(D, self.D_max + 1)
            )
        elif D < 0:
            return 0.0
        else:
            return poisson_pmf(D, self.lambda_)

    def recompensa(self, s, a, s_):
        D = (s + a) - s_

        if s_ == self.s_min:
            #Promedio ponderado de G para todos D que llevan a s_ min o menos
            total_prob = 0.0
            total_reward = 0.0
            for d in range(D, self.D_max + 1):
                p = poisson_pmf(d, self.lambda_)
                total_prob += p
                total_reward += p * self._ganancia(s, a, d)
            return total_reward/total_prob if total_prob > 0 else 0.0
        
        elif D < 0:
            return 0.0
        
        else:
            return self._ganancia(s, a, D)
        
    def es_terminal(self, s):
        #Como un negocio opera indefinidamente, siempre es falso
        return False


if __name__ == "__main__":

    inv = Inventario(gama=0.95, lambda_=4)

    print("Estados:", inv.estados)
    print("D_max  :", inv.D_max)
    print()

    for s in [-10, 0, 5, 20]:
        print(f"  acciones_legales({s:>3}) = 0..{max(inv.acciones_legales(s))}")

    print()

    casos = [
        ( 0, 5,  1),
        ( 5, 3,  4),
        (10, 0,  6),
        (-5, 5, -4),
    ]
    print(f"{'(s, a, s_)':^25} {'T(s,a,s_)':^12} {'R(s,a,s_)':^12}")
    print("-" * 52)
    for (s, a, s_) in casos:
        t = inv.prob_transicion(s, a, s_)
        r = inv.recompensa(s, a, s_)
        print(f"  ({s:>3},{a:>3},{s_:>3})        {t:>10.4f}   {r:>10.2f}")

    print("\n" + "=" * 60)
    print("SEGUNDA PARTE: Iteracion de Valor  (λ=4, γ=0.95)")
    print("=" * 60)

    pi_star, V = iteracion_valor(inv, epsilon=1e-4)

    print("-" * 60)
    print(f"{'Estado':^20}{'Accion (pedir)':^20}{'Valor V(s)':^20}")
    print("-" * 60)
    for s in inv.estados:
        accion = pi_star.get(s, "—")
        print(f"{s:^20}{str(accion):^20}{V[s]:^20.2f}")
    print("-" * 60)

    print("\n" + "=" * 60)
    print("COMPARACION DE POLITICA: λ=4 vs λ=8")
    print("=" * 60)

    inv8 = Inventario(gama=0.95, lambda_=8)
    pi8, V8 = iteracion_valor(inv8, epsilon=1e-4)

    print(f"{'Estado':^12} {'π*(λ=4)':^14} {'π*(λ=8)':^14} {'Diferencia':^12}")
    print("-" * 54)
    for s in inv.estados:
        a4 = pi_star.get(s, 0)
        a8 = pi8.get(s, 0)
        diff = a8 - a4
        signo = f"+{diff}" if diff > 0 else str(diff)
        print(f"{s:^12} {str(a4):^14} {str(a8):^14} {signo:^12}")
    print("-" * 54)


"""
Contesta las preguntas aquí mismo (has espacio entre las preguntas):

1. ¿Cómo se comporta las transiciones y las ganancias para casos específicos de $s$ y $a$? 
Mientras haya mayor inventario disponible, mayor gaancia. El backlog perjudica porque acumulas penalizacion
+ el margen perdido, sin ganar nada en las ventas.
2. ¿Qué psa si hay mucho almacen? 
V(s) sigue creciendo, pero cada vez mas despacio. Tener mas stock es mejor que menos, pero el costo
de almacenamiento tambien puede afectar a la ganancia. 
3. ¿Que pasa si hay muy poco o estamos sin almacen? 
El backlog es la peor situacion, pues no vendes nada, pag penalizacion por cada unidad faltante
y pierdes el margen.
4. ¿Existe un punto donde la ganancia sea máxima?  
Sí, I = 9. 
---
5. ¿Cómo se ve la política óptima? ¿Tiene sentido?
Si el inventario es menor o igual a 5, pedir hasta llegar a 9, si no, no pedir.
Tiene sentido, ya que el costo fijo de $40 hace que no valga la pena ordenar cantidades
pequeñas.
6. ¿Como se comporta la función de valor de estado V(s)?
Para s igual o menor a 5, V sube exactamente $80 por unidad, que es exactamente el costo de compra.
Para s mayor o igual a 6, V sigue creciendo pero con incrementos irregulares y decrecientes.
7. ¿Cómo cambiaría la política si la variabilidad de la demanda (lambda) aumenta de 4 a 8?
El nivel oprimo sube de 9 a 13 y el punto de reorden sube de 5 a 10. Necesitas mas colchon
para evitar el backlog.
"""