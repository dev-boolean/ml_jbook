# Conclusiones y reflexión crítica

## ¿Qué entorno fue más rápido y por qué?

**scikit-learn ganó en las tres etapas medidas**, y por márgenes muy
distintos según la etapa:

| Etapa | scikit-learn | PySpark | PySpark / sklearn |
|---|---|---|---|
| Preprocesamiento | 4.6s | 68.2s | **14.8×** más lento |
| Entrenamiento (con CV) | 18.2 min | 40.9 min | **2.2×** más lento |
| Predicción | 2.6s | 36.0s | **13.9×** más lento |

El patrón es revelador: la brecha es enorme (≈15×) en preprocesamiento y
predicción, pero se reduce a 2.2× en entrenamiento. Eso es exactamente lo que
predice la arquitectura de Spark: preprocesamiento y predicción son etapas
"ligeras" (transformaciones relativamente baratas por fila) donde el costo
**fijo** de Spark —arrancar la JVM, planificar el DAG, serializar tareas,
coordinar el shuffle— domina sobre el trabajo real. En entrenamiento, en
cambio, hay mucho más trabajo genuino por hacer (construir árboles profundos
sobre 1.8M filas, 18 veces para el grid × folds), así que ese mismo costo
fijo pesa proporcionalmente menos frente al cómputo real.

La causa de fondo es que Spark está diseñado para **distribuir** trabajo
entre múltiples máquinas de un clúster; corriendo en `local[*]` sobre una
sola máquina, se paga todo el overhead de coordinación de un sistema
distribuido sin ninguno de sus beneficios (no hay más máquinas entre las
cuales repartir el trabajo). scikit-learn + NumPy, en cambio, corre en un
solo proceso con rutinas en C/Cython ya optimizadas para memoria compartida,
sin cruzar nunca la frontera JVM↔Python (py4j) que sí paga PySpark en cada
operación.

## ¿Cuál fue más preciso?

| Métrica | scikit-learn | PySpark |
|---|---|---|
| Accuracy | 0.6605 | 0.6354 |
| Precision | 0.2085 | 0.1988 |
| Recall | 0.6642 | 0.6849 |
| F1 | 0.3173 | 0.3082 |
| ROC AUC | 0.7245 | 0.7177 |

Con la misma metodología, **los dos frameworks quedan prácticamente
empatados** — sklearn gana por poco en accuracy/precision/ROC AUC, PySpark
gana por poco en recall. Las diferencias (unos pocos puntos porcentuales) son
del orden esperable entre dos implementaciones distintas de RandomForest:
scikit-learn busca el split óptimo exacto por nodo, Spark ML usa un método de
histogramas/bins pensado para escalar a datos distribuidos; además cada
framework hizo su propio split 80/20 de forma independiente, así que ni
siquiera son exactamente las mismas filas de test.

Un punto metodológico central para ambos modelos es el manejo del desbalance
de clases (~12% son `default`). scikit-learn lo resuelve con
`class_weight="balanced"` en el constructor de `RandomForestClassifier`;
`pyspark.ml.classification.RandomForestClassifier` no tiene un parámetro
equivalente, así que los pesos se calculan manualmente con la misma fórmula
que usa sklearn internamente (`n_muestras / (n_clases × n_c)`) y se pasan vía
`weightCol`. Sin este ajuste, un RandomForest entrenado sobre datos así de
desbalanceados puede terminar prediciendo casi exclusivamente la clase
mayoritaria — con una accuracy engañosamente alta (~88%, el porcentaje de la
clase 0) y un ROC AUC todavía razonable (el AUC mide qué tan bien se
*ordenan* las probabilidades, no si el modelo clasifica bien a un umbral
fijo), pero con precision/recall/F1 cercanos a cero para la clase que
realmente importa. Por eso el desempeño de un modelo de riesgo crediticio
debe evaluarse con F1/recall/precision de la clase minoritaria, y no solo con
accuracy o ROC AUC.

## ¿A partir de qué volumen de datos PySpark comienza a superar a scikit-learn?

Como el proyecto exige el dataset completo sin muestreo, esta curva de cruce
no se midió directamente. Pero los tiempos medidos sí dan una pista: el
overhead fijo de Spark (~64s extra en preprocesamiento, ~34s extra en
predicción, independientemente del tamaño de esas operaciones dentro del
rango de este dataset) es sustancial y en gran parte **no depende de cuántas
filas se procesen** — es el costo de arrancar la JVM, planificar tareas y
cruzar la frontera py4j. Para que ese costo fijo deje de dominar, el trabajo
real tendría que crecer varios órdenes de magnitud.

Más importante aún: `local[*]` corre sobre **una sola máquina**. La ventaja
real de Spark no es "más CPUs" sino *distribuir* datos que no caben en la RAM
o el disco de una sola máquina entre varias. Con 2.26M filas (~1.7GB), este
dataset cabe cómodo en la RAM de casi cualquier laptop moderna — scikit-learn
+ pandas nunca sienten presión de memoria. La literatura y los benchmarks
comunitarios sobre Spark en modo local sugieren que el cruce real (donde
Spark empieza a ganar) suele aparecer recién en el orden de decenas de GB, y
casi siempre requiere un clúster de varios nodos reales, no una sola máquina
con más núcleos. Este proyecto está, por diseño (dataset completo obligatorio
pero de tamaño moderado), firmemente del lado donde scikit-learn gana.

## ¿Qué aporta LIME y cuáles son sus limitaciones en entornos distribuidos?

LIME aportó lo que ninguna métrica agregada puede dar: una explicación *por
instancia* de por qué el modelo se equivocó en un caso puntual, en términos
de qué variables empujaron la predicción en cada dirección — accionable para
un analista de crédito revisando un caso específico, no solo para reportar
un número de desempeño global.

La limitación central es que LIME es **intrínsecamente local** (perturba con
NumPy y ajusta una regresión con scikit-learn, todo en el proceso Python del
driver) y no tiene ningún concepto de DataFrame distribuido. Explicar una
predicción de un modelo Spark exige un puente que, para cada lote de
perturbaciones, haga `createDataFrame` → `VectorAssembler.transform` →
`model.transform` → `collect`. El costo es medible y no trivial: cada
`explain_instance` contra el modelo de PySpark tardó **~39-40 segundos**
(incluso con `num_samples=500`, ya reducido desde el default de 5,000,
precisamente para acotar ese costo), contra una respuesta prácticamente
instantánea del lado de scikit-learn, donde `predict_proba` nunca sale del
proceso Python.

Dos limitaciones adicionales, ya señaladas en `06_interpretability_lime.ipynb`:
los nombres de las variables categóricas de Spark llegan degradados
(`cat_ohe_4`, `cat_ohe_213`, ...) porque `pyspark.ml`'s `OneHotEncoder` no
expone `get_feature_names_out()` como sklearn; y el fondo estadístico del
explicador de Spark usa una muestra de 5,000 filas en vez del `X_train`
completo que sí usa el lado de scikit-learn — una limitación honesta más que
un defecto, porque escalar LIME al dataset completo no tendría mucho sentido
práctico de todas formas (el valor de LIME es explicar una predicción
puntual, no describir el dataset entero).

## ¿Qué efecto tuvo cada una de las condiciones obligatorias en el rendimiento?

- **Dataset completo sin muestreo** (2,260,701 filas → ~1.81M train / ~452K
  test en ambos frameworks): garantiza que la comparación de tiempos y
  métricas ocurre a una escala real, no de juguete — pero también es
  precisamente la razón por la que este proyecto se queda del lado donde
  scikit-learn gana en velocidad (ver pregunta 3): a esta escala, el
  overhead fijo de Spark nunca llega a amortizarse.
- **`.persist(StorageLevel.MEMORY_AND_DISK)` obligatorio**: crítico para que
  el `CrossValidator` no vuelva a computar el pipeline de preprocesamiento en
  cada uno de sus 18 ajustes (9 combinaciones × 2 folds) — sin persistir,
  Spark recalcularía el DAG completo de lectura+limpieza+encoding cada vez
  por su evaluación *lazy*.
- **`CrossValidator` + `ParamGridBuilder` (no loops manuales)**: obliga a
  pagar la maquinaria completa de Spark ML para la búsqueda de
  hiperparámetros en vez de un bucle ligero como en sklearn — parte de por
  qué el entrenamiento en Spark, aun siendo la etapa donde la brecha
  relativa es menor, sigue tardando 2.2× más.
- **Config de memoria/paralelismo ajustada a los recursos reales de la
  máquina** (en vez de los `8g`/`400` particiones literales del enunciado):
  `shuffle.partitions` y el paralelismo de `local[*]` escalan con los
  núcleos disponibles, y la memoria del driver tiene que escalar en esa
  misma proporción — más núcleos habilitan más tareas de entrenamiento
  concurrentes, y cada una necesita su propio espacio de heap. Tratar
  paralelismo y memoria como parámetros independientes, en vez de ajustarlos
  juntos según el hardware real, es la forma más común de subestimar cuánta
  memoria necesita en realidad una sesión de Spark local.
