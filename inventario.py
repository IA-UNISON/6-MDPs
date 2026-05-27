from MDPs import MDP, iteracion_valor
from math import exp, factorial

class Inventario(MDP):
    """
    Clase que representa un MDP para el problema del camión mágico.
    
    Si caminas, avanzas 1 con coso 1
    Si usas el camion, con probabilidad rho avanzas el doble de donde estabas
    y con probabilidad 1-rho te quedas en el mismo lugar. Todo con costo 2.
    
    El objetivo es llegar a la meta en el menor costo posible
    
    """    
    
    def __init__(self, gamma,lamb,cap_max,cap_min,precio_venta,costo_per_unidad,costo_fijo_pedido,costo_almacenamiento,costo_backlogging ): #TODO: Agregar lo que se requiera

        self.gamma = gamma
        self.lambda_ = lamb
        self.cap_max = cap_max
        self.cap_min = cap_min
        self.precio_venta = precio_venta
        self.costo_per_unidad = costo_per_unidad
        self.costo_fijo_pedido = costo_fijo_pedido
        self.costo_almacenamiento = costo_almacenamiento
        self.costo_backlogging = costo_backlogging
        estados = list(range(self.cap_min, self.cap_max + 1))
        super().__init__(estados, gamma)

    def acciones_legales(self,s):
        return list(range(0, self.cap_max - s + 1))

    def recompensa(self, s, a, s_):
        inventario_disponible = s + a

        demanda_real = inventario_disponible - s_

        vendidas = min(demanda_real, inventario_disponible)

        ingresos = self.precio_venta * vendidas

        costo_compra = self.costo_per_unidad * a

        costo_fijo = self.costo_fijo_pedido \
            if a > 0 else 0

        exceso = max(0, s_ - self.cap_min)

        costo_almac = self.costo_almacenamiento * max(0, s_)


        deuda = max(0, -s_)

        costo_backlog = self.costo_backlogging * deuda
        recompensa = ingresos - costo_compra - costo_fijo - costo_almac - costo_backlog

        return recompensa


    def prob_transicion(self, s, a, s_):
        """
        s: estado actual (inventario al inicio del día)
        a: accion (unidades a pedir)
        s_: estado siguiente (inventario al final del día)
        """

        inventario_disponible = s + a

        demanda_real = inventario_disponible - s_

        if demanda_real < 0:
            return 0

        probabilidad = (self.lambda_ ** demanda_real * exp(-self.lambda_)) / factorial(demanda_real)

        return probabilidad
                
    def es_terminal(self, s):
        return False


if __name__ == "__main__":
    inventario = Inventario(
        0.95,
        4,
        20,
        -10,
        150.00,
        80.00,
        40.00,
        5.00,
        15.00
    )

    pi_star, V = iteracion_valor(inventario, 1e-4)
    print("-" * 60)
    print("Estado".center(20) + "Acción".center(20) + "Valor".center(20))
    print("-" * 60 )
    for s in pi_star:
        print(f"{s:^20}{pi_star[s]:^20}{V[s]:^20.2f}")
    print("-" * 60)


"""
Contesta las preguntas aquí mismo (has espacio entre las preguntas):

1. ¿Cómo se comporta las transiciones y las ganancias para casos específicos de s y a? 

Las transiciones siguen una distribución de Poisson con λ=4. Para un estado s con acción a:
- Inventario disponible: s + a
- El proximo estadi seria el inventario disponible menos la demanda 
- La demanda esperada es 4 unidades/día

Las ganancias varían según el inventario:
- Con s=-10, a=19: con inventario disponible = 9 unidades. Si demanda=4, s'=5. Los costos de backlogging (-15/u) son altos
  por la deuda anterior, pero se recupera con ingresos (150*min(4,9)=600) menos costos.

2. ¿Qué psa si hay mucho almacen? 

Si hay mucho almacén (s ≥ 6), la política óptima es NO PEDIR (a=0). Esto porque:
- El costo fijo de pedido ($40) es muy alto a comparacion con el del  beneficio de tener más stock
- La demanda esperada es 4 unidades, así que con mas de 6 unidades  ya se cubre la demanda 

3. ¿Que pasa si hay muy poco o estamos sin almacen? 

Si hay poco inventario (s ≤ 5), la política es pedir mucho:
- s=-10: pedir 19 unidades (máximo posible, llega a 9)
- s=0: pedir 9 unidades (llega a 9)
- s=5: pedir 4 unidades (llega a 9)

4. ¿Existe un punto donde la ganancia sea máxima?  

No existe un punto unico donde la ganancia sea maxima. La función de valor V(s) crece continuamente 
desde s=-10 hasta s=20.
- De s=-10 a s=5: valor creciente 
  Diferencia: ΔV = 4512.21 - 3312.21 = 1200 (por 15 unidades)

---

5. ¿Cómo se ve la política óptima? ¿Tiene sentido?
Una politica optima seria una en la que 
- Si s ≤ 5: PEDIR de forma decreciente  hasta llegar a 9 unidades
- Si s ≥ 6: NO PEDIR

Es optima por que 
- Es conservadora cuando hay poco stock (riesgo alto de pérdidas por escasez)
- Es ahorradora cuando hay mucho stock (riesgo bajo, costo fijo no se justifica)
- Minimiza el costo total de operación a largo plazo

La política es MONÓTONA: conforme aumenta s, la acción a disminuye (menos necesidad de pedir).

6. ¿Como se comporta la función de valor de estado V(s)?
- Crecimiento: V(s) aumenta de 3,312.21 (s=-10) a 5,632.68 (s=20)
el crecimiento es mayor en los primeros estados (recuperación de deuda)


Cambios V(s+1) - V(s):
- De s=-10 a s=5: ~ΔV = 80 por unidad
- De s=6 a s=20: ~ΔV = 76 a 60 por unidad 


7. ¿Cómo cambiaría la política si la variabilidad de la demanda (lambda) aumenta de 4 a 8?

Si λ aumenta de 4 a 8:
- La demanda esperada sería el DOBLE (8 unidades/día vs 4)
- El RIESGO de quedarse sin stock aumenta
- Con mayor variabilidad, es más costoso quedarse sin stock, así que la política
se vuelve más conservadora 

"""