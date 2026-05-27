![](ia.png)


## Un problema de inventario

### Contexto del Problema

La empresa *Necroelectronica* distribuye un componente electrónico especializado. Debido a la naturaleza del producto, el gerente de inventario debe decidir cada tarde cuántas unidades pedir al proveedor para maximizar el beneficio a largo plazo.

### Dinámica de Operación

El ciclo de operación sigue una secuencia estrictamente temporal:

1. **Mañana:** Se reciben las unidades que fueron pedidas la tarde anterior (el tiempo de entrega es de 1 día). Estas se incorporan inmediatamente al inventario disponible.
2. **Durante el día:** Ocurre la demanda de los clientes.
3. **Tarde:** Se contabiliza el inventario final, se pagan los costos de almacenamiento o penalizaciones, y se decide la cantidad a pedir para recibir el día siguiente.

### Parámetros y Valores Realistas

Para el modelado, considera lo siguiente:

* **Capacidad de Almacén:** El estante tiene una capacidad máxima de **20 unidades**. No se pueden almacenar más unidades.
* **Comportamiento de la Demanda ($D$):** La demanda diaria sigue una **Distribución de Poisson** con una media de $\lambda = 4$ unidades/día, utilizando la función de probabilidad de Poisson: $$f(k; \lambda) = \frac{e^{-\lambda} \lambda^k}{k!}$$
* **Estructura de Costos y Ganancias:**
  * **Precio de Venta:** $150.00 por unidad vendida.
  * **Costo de Compra:** $80.00 por unidad pedida al proveedor.
  * **Costo Fijo de Pedido:** $40.00 por cada pedido realizado (independientemente de la cantidad, siempre que sea $> 0$).
  * **Costo de Almacenamiento:** $5.00 por cada unidad que se quede en el estante al final del día.
  * **Costo de Backlogging (Inventario Negativo):** Si la demanda supera las existencias, los clientes aceptan esperar, pero la empresa incurre en un costo de "buena voluntad" y logística de **$15.00 por unidad faltante** al final del día.
  * **Pérdida por Demanda no Satisfecha:** Además del costo de backlogging, cada unidad demandada que no puede entregarse en el momento representa una pérdida de oportunidad (margen no ganado).
* **A tomar en cuenta** 
  * **Límite Inferior de Inventario (Backlogging):** Aunque la capacidad máxima es 20, permitiremos un backlog de hasta -10 unidades para modelar la demanda insatisfecha. Por lo tanto, el espacio de estados es $S = \{-10, -9, \dots, 20\}$.
  * **Función de Recompensa ($R$):** Construye la ecuación de recompensa inmediata $R(s, a)$ que incluya los ingresos por ventas, los costos de pedido (fijos y variables), los costos de mantenimiento y las penalizaciones por faltantes.
  * **Factor de Descuento ($\gamma$):** Establecemos $\gamma = 0.95$. Esto representa la preferencia por beneficios inmediatos y garantiza la convergencia de la serie.
  * **Umbral de Convergencia ($\epsilon$):** El algoritmo se detendrá cuando el cambio máximo en la función de valor sea $\max_{s \in S} |V_{k+1}(s) - V_k(s)| < 10^{-4}$.
  * Recuerda que el inventario disponible para satisfacer la demanda del día $t$ es la suma del inventario que quedó al final del día $t-1$ más el pedido realizado esa misma tarde.
  * Para efectos de cómputo, puedes truncar la distribución de Poisson en un valor donde la probabilidad acumulada sea cercana a 1.


### Primera parte: Modelo MDP (40 puntos)

A partir del escenario anterior, Completa la clase inventario en el archivo `inventario.py` y su instanciación en un caso particular como el que definimos. Procura hacerlo de forma genérica en cuanto a ganancias, costos y valores.

Contesta lo siguiente

1. ¿Cómo se comporta las transiciones y las ganancias para casos específicos de $s$ y $a$? 
2. ¿Qué psa si hay mucho almacen? 
3. ¿Que pasa si hay muy poco o estamos sin almacen? 
4. ¿Existe un punto donde la ganancia sea máxima?  

### Segunda parte: Programación dinámica (60 puntos)

Para resolver el problema, utiliza el método de iteración de valores en el archivo `inventario.py` y contestalas siguientes preguntas:

1. ¿Cómo se ve la política óptima? ¿Tiene sentido?
2. ¿Como se comporta la función de valor de estado $V(s)$?
3. ¿Cómo cambiaría la política si la variabilidad de la demanda ($\lambda$) aumenta de 4 a 8?

