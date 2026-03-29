## Prueba con diferentes valores de rho. ¿Qué observas? ¿Porqué crees que pase eso?
Con gamma = 0.999
Para valores muy bajos de rho, SARSA tiende a explorar más tramos donde usar el camión y Q-Learning tiende a ser mucho más conservador con los tramos donde se utiliza le camión.
Para valores medios de rho podemos observar comportamientos similares, SARSA explora bastante y Q-Learning no tanto
Para valores altos de rho las politicas terminan siendo muy parecidas, encuentran casi los mismos tramos pero SARSA tiende a encontrar mas tramos en donde se puede usar el camión

## Prueba con diferentes valores de gama. ¿Qué observas? ¿Porqué crees que pase eso?
Con rho = 0.9
Para valores bajos y medios de gamma, las políticas son muy parecidas o completamente iguales, ambas son conservadoras en los
tramos en los cuales se debe usar el camión y coinciden mucho.

Para valores altos de gamma empiezan a diferir un poco las políticas, SARSA se vuelve mucho más exploradora
y tiende a encontrar muchos mas tramos. Por otro lado, Q-Learning tiende a ser algo más conservadora,
también obtiene algunos tramos más pero se queda bastante parecida

## ¿Qué tan diferente es la política óptima de SARSA y Q-learning?
Ambas suelen coincidir en muchos tramos independientemente de los valores de rho y gamma, sin embargo en valores
altos de ambos, es decir, si la probabilidad de éxito es alta y además valoramos mucho los éxitos futuros,
las políticas difieren un poco más, SARSA identifica muchos más tramos que Q-Learning. 
Esto tiene sentido porque SARSA es on-policy (aprende la política que ejecuta, que es más conservadora por el epsilon), 
mientras que Q-learning es off-policy y aprende la política óptima independientemente de lo que hace.
Esto hace que Q-learning sea un poco más "decidido" y SARSA más prudente.

## ¿Cambia mucho el resultado cambiando los valores de recompensa?
Si, cambia bastante
### Recompensa normal:

Los tramos donde se debe usar el camión segun SARSA son:
[2, 4, 9, 18, 34, 35, 36, 72]
Los tramos donde se debe usar el camión segun Qlearning son:
[2, 4, 9, 18, 36, 72]

### Vamos a recompensar más el llegar a la meta (300 en vez de 100):
Los tramos donde se debe usar el camión segun SARSA son:
[1, 4, 8, 9, 17, 18, 35, 36, 72]
Los tramos donde se debe usar el camión segun Qlearning son:
[2, 4, 9, 17, 18, 30, 36, 72]

Aqui ahora podemos observar que Q-Learning empieza a encontrar muchos más tramos para usar el camión, mientras
que SARSA se queda bastante parecida

### Vamos a recompensar igual el llegar pero penalizar menos el pasarse (-30 en vez de -100)

Los tramos donde se debe usar el camión segun SARSA son:
[1, 2, 4, 9, 17, 18, 35, 36, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82]
Los tramos donde se debe usar el camión segun Qlearning son:
[2, 3, 4, 9, 18, 36, 72, 73]

Podemos observar que ahora identifican muchos más tramos que antes, especialmente SARSA

### Vamos a penalizar más el usar el camión (-10 en vez de -2)

Los tramos donde se debe usar el camión segun SARSA son:
[17, 18, 35, 36, 72]
Los tramos donde se debe usar el camión segun Qlearning son:
[9, 18, 36, 72]

Ambas políticas se vuelven mucho más conservadoras y no encuentran muchos tramos para usar el camión 

### Vamos a penalizar más el caminar que usar el camión (-3 para caminar y -2 para el camión)

Los tramos donde se debe usar el camión segun SARSA son:
[2, 4, 9, 18, 35, 36, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93]
Los tramos donde se debe usar el camión segun Qlearning son:
[2, 4, 9, 18, 36, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]

Ambas políticas empiezan a usar mucho más el camión ahora y SARSA lo usa en la mayoría de posiciones

Vamos a probar este enfoque pero ahora penalizando MUCHO el pasarse

### Además de penalizar más caminar que usar el camión, vamos a penalizar el pasarse (-300 en vez de -100)

Los tramos donde se debe usar el camión segun SARSA son:
[1, 2, 4, 9, 17, 18, 35, 36, 71, 72]
Los tramos donde se debe usar el camión segun Qlearning son:
[1, 2, 4, 9, 18, 36, 72]

Podemos observar que a pesar de que caminar sea más penalizado que usar el bus, como es mucho más penalizado
pasarse de la meta, la política vuelve a ser más conservadora y prefiere caminar a usar el bus en la mayoría del 
trayecto, pareciendose más a nuestra recompensa inicial

## ¿Cuantas iteraciones se necesitan para que funcionen correctamente los algoritmos?
### Normal
n_ep = 100,000
SARSA_iter = 50
Q-Learning_iter = 1000
Los tramos donde se debe usar el camión segun SARSA son:
[1, 2, 4, 8, 9, 17, 18, 35, 36, 72]
Los tramos donde se debe usar el camión segun Qlearning son:
[2, 4, 9, 18, 36, 72]

### Igualamos SARSA a Q-Learning
n_ep = 100,000
SARSA_iter = 1000
Q-Learning_iter = 1000
Los tramos donde se debe usar el camión segun SARSA son:
[2, 4, 8, 9, 17, 18, 35, 36, 72]
Los tramos donde se debe usar el camión segun Qlearning son:
[2, 4, 9, 18, 36, 72]

SARSA empieza a explorar muchos más tramos

### Reducimos los episodios a la mitad
n_ep = 50,000
SARSA_iter = 50
Q-Learning_iter = 1000
Los tramos donde se debe usar el camión segun SARSA son:
[2, 4, 9, 18, 35, 36, 72]
Los tramos donde se debe usar el camión segun Qlearning son:
[1, 2, 4, 9, 17, 18, 36, 70, 72]

Notamos que Q-Learning empieza a tener muchos más valores que SARSA 

### Reducimos los episodios a 10,000
n_ep = 10,000
SARSA_iter = 50
Q-Learning_iter = 1000
Los tramos donde se debe usar el camión segun SARSA son:
[2, 4, 9, 17, 18, 36, 72]
Los tramos donde se debe usar el camión segun Qlearning son:
[2, 4, 9, 18, 25, 32, 36, 68, 70, 72]

SARSA deja de capturar tantos tramos y se vuelve más conservador, mientras Q-Learning captura muchos más 
tramos donde usar le bus
