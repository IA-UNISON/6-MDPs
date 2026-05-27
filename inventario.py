"""
Para desarrollar el problema del inventario.

"""

from MDPs import MDP, iteracion_valor
import math

class Inventario(MDP):
    
    def __init__(self, lambda_, gama):
        """
        Tupla de estados, promedio de venta, factor de descuento 
        """
        self.estados = tuple(range(-10, 21))
        self.lambda_ = lambda_
        self.gama = gama
    
    def acciones_legales(self, s):
        """
        Que tantas unidades pido en la tarde ?
        """
        return list(range(0, 21 - s))
    
    def recompensa(self, s, a, s_):
        """
        Le resto todos los diferentes costos a los ingresos.
        """
        x = s + a
        d = x - s_

        ventas = min(x, d) if x > 0 else 0
        unmet = max(d - x, 0)

        ingresos = 150 * ventas
        costo_compra = 80 * a
        costo_fijo = 40 if a > 0 else 0
        holding = 5 * max(s_, 0)
        backlog = 15 * max(-s_, 0)

        R = ingresos - costo_compra - costo_fijo - holding - backlog
        return R

        
    def prob_transicion(self, s, a, s_):
        """
        s_ es el inventario disponible después de la demanda del día. Si la demanda es tan grande
        que el backlog cae por debajo de -10, lo agregamos al estado límite s_ = -10.
        """
        x = s + a
        if s_ > x or s_ < -10 or s_ > 20:
            return 0.0

        def poisson(k):
            if k < 0:
                return 0.0
            return math.exp(-self.lambda_) * (self.lambda_ ** k) / math.factorial(k)

        if s_ == -10:
            demanda_min = x + 10
            prob = 0.0
            d = demanda_min
            while True:
                p = poisson(d)
                if p < 1e-12:
                    break
                prob += p
                d += 1
                if d > demanda_min + 100:
                    break
            return prob

        d = x - s_
        return poisson(d)
                
    def es_terminal(self, s):
        """
        No hay un estado terminal.
        """
        return False


if __name__ == "__main__":
    lambda_param = 4.0
    gama = 0.95
    epsilon = 1e-4
    max_iter = 1000

    inventario = Inventario(lambda_param, gama)
    pi_star, V = iteracion_valor(inventario, epsilon=epsilon, max_iter=max_iter, debug=True)

    print("-" * 60)
    print("Estado".center(20) + "Acción".center(20) + "Valor".center(20))
    print("-" * 60 )
    for s in sorted(pi_star.keys()):
        print(f"{s:^20}{pi_star[s]:^20}{V[s]:^20.2f}")
    print("-" * 60)


"""
Contesta las preguntas aquí mismo (has espacio entre las preguntas):

1. ¿Cómo se comporta las transiciones y las ganancias para casos específicos de $s$ y $a$? 
    las transiciones dependen de cuanto sea la demanda aleatoria el siguiente dia. el inventario al dia siguiente es primero
    la suma de s + a, a lo que luego se le resta la demanda para determinar s_. si la demanda es muy alta podemos tener inventario
    negativo.

2. ¿Qué psa si hay mucho almacen?
    El costo del almacen se vuelve muy alto y la politica correcta en esos casos, como muestran los resultados, es no pedir mas unidades.

3. ¿Que pasa si hay muy poco o estamos sin almacen?
    Si hay muy poco o tenemos inventario negativo la politica correcta es pedir un mayor numero de unidades, ya que en la implementacion 
    tambien estamos penalizando el backlog y la perdida de oprtunidades

4. ¿Existe un punto donde la ganancia sea máxima?  
    segun los resultados obtenidos es optimo mantenernos entre el rango de 6 a 9 unidades aproximadamente.

---

5. ¿Cómo se ve la política óptima? ¿Tiene sentido?
    Tiene sentido y es lo esperado (cuando hay bajo inventario se piden mas undidades, cuando hay mucho inventario se piden menos),
    lo unico que no tiene sentido para mi es el salto brusco entre 5 y 6 en el que pasamos de pedir 4 unidades a 0. Tal vez algo 
    hice mal en la implementacion pero el resto de la politica tiene sentido.

6. ¿Como se comporta la función de valor de estado V(s)?
    Es una funcion creciente, mientras mas inventario mas valor esperado tenemos.

7. ¿Cómo cambiaría la política si la variabilidad de la demanda (lambda) aumenta de 4 a 8?
    Simplemente pediriamos mas unidades en la mayoria de los casos para poder satisfacer la mayor demanda promedio.

"""