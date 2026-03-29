## ¿Qué pasa si se modifica el valor de epsilón de la política epsilon-greedy?
Valores bajos (0.001 - 0.05)

SARSA y Q-Learning tienden a recorrer caminos ya conocidos repetidamente y tienden a
quedarse en ciertos patrones de comportamientos

- SARSA: Tiende a ser mas conservador y no recorrer frecuentemente rutas riesgosas
aunque estas presenten una recompensa potencial jugosa
- Q-Learning: Se suele concentrar en rutas que maximizan su recompensa sin tomar
en cuenta el riesgo

Valores medios (0.1 - 0.3)
Ambos algoritmos suelen escapar más fácil de los óptimos locales
- SARSA: Explora más posibilidades pero manteniendo las rutas seguras
- Q-Learning: Aunque distribuye mucho más sus visitas, sigue priorizando
las mayores recompensas potenciales

Valores altos (0.4 - 0.5)
Ahora una gran cantidad de estados suelen ser visitados múltiples veces por 
SARSA y Q-Learning. Sin embargo el comportamiento se vuelve un poco inestable al
subir tanto el epsilon pues subimos mucho la probabilidad de exploración.

- SARSA: Pierde lo conservador y empieza a realizar muchas más acciones aleatorias por
la exploración forzada.
- Q-Learning: Mantiene la tendencia de visitar los estados de alta recompensa, pero
no es tan frecuente

* Conclusión: A bajos valores de epsilon ambos algoritmos tienden a ser más conservadores con
la exploración aunque Q-Learning suele ser un poco más atrevido con las rutas
para maximizar su recompensa. Cuando vamos aumentando el valor podemos observar
que se empieza a salir de los óptimos locales debido a la exploración, sin embargo
el subir más esta probabilidad empieza a mostrar comportamientos erráticos.

## ¿Para que sirve usar una politica epsilon-greedy?
Las políticas de este estilo sirven para mejorar nuestros resultados, pues los algoritmos
tienden a irse al óptimo más cercano (lo cual puede no ser siempre el mejor caso) y 
la forma de evitar esto es, de forma aleatoria con probabilidad $\epsilon$, tomar una acción
sin seguir la política.
Además de esto nos sirven para adaptarnos a entornos de los cuales tenemos conocimiento
incompleto.

## ¿Qué pasa con la política óptima y porqué si p_h es mayor a 0.5?
Siendo p_h la probabilidad de éxito, el aumentar su valor hace cambiar considerablemente
los patrones de las políticas.
Podemos identificar regiones en donde estos 2 algoritmos empiezan a actuar notablemente
distinto a lo usual. 

Zona 1 (30-70):
A medida que vamos aumentando la probabilidad de éxito, los algoritmos suelen tomar 
decisiones mucho más arriesgadas por esta zona, siendo Q-Learning el más arriesgado

Zona 2 (80-99):
Si ph es muy alto, Q-Learning empieza a mostrar un comportamiento menos arriesgado
y mucho más conservador, es decir, empieza siendo muy arriesgado para maximizar
la recompensa y cuando se encuentra cerca del final, se vuelve mucho más sistemático

## ¿Y si es 0.5?
En el caso de $ph=0.5$ SARSA se muestra más arriesgada, apostando en mucho más a partir
de 30-35 y Q-Learning es muchisimo más conservador, haciendo la apuesta mínima en la mayoría
de los casos.
Esto es por la naturaleza de ambas políticas, Q-Learning al valorar las acciones
óptimas futuras, tiene una desconexión entre su exploración y su actualización de valores.
Sin embargo SARSA no tiene estas carencias y por eso se suele ver mucho más que SARSA
tiende a explorar en este caso

## ¿Y si es menor a 0.5?
Podemos observar, que cuando ph es muy pequeño (0.1 - 0.4) los comportamientos se comienzan
a volver erráticos, no tenemos regiones muy claras de donde se producen apuestas realmente
y suelen ser bastante dispersas, siguiendo un comportamiento de apostar poco y subsecuentemente
apostar una cantidad grande (20-50). A medida que aumentamos nuestro ph, las apuestas comienzan
a ser menos erráticas pero siguen siendo grandes.
SARSA, en todos los valores de ph menores a 0.5, se muestra mucho más atrevido que Q-Learning,
haciendo muchas apuestas grandes de formas consecutivas a diferencia de Q-Learning que se mantiene
regularmente haciendo apuestas mínimas y ocasionalmente haciendo apuestas grandes de la nada.

## ¿Qué pasa si se modifica el valor de la tasa de aprendizaje?
Cuando usamos un alfa bajo, observamos un comportamiento muy conservador, manteniendo apuestas bajas
pero siendo SARSA algo mas atrevido en estos casos, haciendo apuestas un poco más grandes
en algunos momentos. También podemos observar que se muestra un patrón de apuestas
adyacente, es decir, si apuesta un poco más del mínimo crece de poco a poco hasta llegar
a un valor alto y luego decrece de manera rápida.
Cuando aumentamos el valor de alfa podemos ver algo interesante y es que SARSA se vuelve
un tanto mas errático, esto es debido a como actúa al momento del aprendizaje, pues las accionones
exploratorias hacen cambios drásticos en forma de efecto dominó, por otro lado, Q-Learning no se
ve tan afectado por la forma en la que aprende, pues este siempre ve la mejor acción posible en el
siguiente estado, independientemente de la acción que de verdad vaya a tomar, esto 
actúa como un filtro natural contra cambios muy drásticos, manteniendo estabilidad.


## ¿Qué pasa si se modifica el valor de gama?
Al reducir el factor de descuento a valores muy chicos como 0.10, ambas políticas se vuelven
extremadamente erráticas, empezando a hacer apuestas muy altas de forma constante, pero
volviendose más conservadores al final.
A medida que vamos aumentando el valor de gama, podemos observar como se va reduciendo
la consistencia con la que se hacen apuestas altas, optando en bastantes más ocasiones
por apuestas mínimas o bajas. Q-Learning particularmente adopta este patrón de apuestas mínimas
y subsecuentemente apuestas altas que luego bajan, mientras que SARSA se mantiene medianamente
consistente haciendo apuestas moderadas-altas.
