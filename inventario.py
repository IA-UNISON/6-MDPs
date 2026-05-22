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
        pass

    def es_terminal(self, s):
        pass