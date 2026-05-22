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
        pass
    
    def prob_transicion(self, s, a, s_):
        pass
                
    def recompensa(self, s, a, s_):
        pass

    def es_terminal(self, s):
        pass