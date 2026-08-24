# Predicción de Default en Préstamos — Lending Club

**scikit-learn vs PySpark: rendimiento, precisión e interpretabilidad**

Autores: Santiago Hurtado, Juan Marín, Andrés Parejo

## Objetivo

Construir un modelo de clasificación supervisada que prediga si un préstamo
emitido por Lending Club resultará en *default* (`1`, Charged Off) o será
pagado completamente (`0`), comparando el desempeño de scikit-learn y PySpark
sobre el **dataset completo**, sin muestreo ni reducción de filas, e
interpretando las predicciones con LIME.

## Dataset

- **Fuente:** [Lending Club Loan Data — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Archivo:** `accepted_2007_to_2018Q4.csv` — **2,260,701 préstamos**, 151
  columnas originales (cubre 2007–2018 Q4; es el archivo real disponible bajo
  este dataset de Kaggle, ligeramente distinto al rango 2007–2020 mencionado
  en el enunciado, pero supera ampliamente el mínimo exigido de 1.3M filas).
- Se usan las 2,260,701 filas en **todo** cálculo obligatorio (estadísticas,
  pruebas de hipótesis, entrenamiento). Solo se muestrea donde el propio
  enunciado lo permite explícitamente (pairplot / scatter matrix), y en
  visualizaciones de densidad (violin/KDE) donde graficar cada punto no
  aportaría información adicional — ambos casos están señalados explícitamente
  donde ocurren.

## Entornos de ejecución

Los notebooks `00`–`02`, `04` y `06` corrieron primero en **Google Colab**
(sin JVM local disponible en ese momento). Ahí surgió un problema real: el
`RandomForestClassifier` de PySpark en `05_modeling_spark.ipynb` se entrenó
sin ningún manejo de desbalance de clases y terminó prediciendo "no default"
para el 100% del test set (precision/recall/F1 = 0.0, a pesar de un ROC AUC
de 0.71 nada despreciable — la métrica de selección del `CrossValidator`
no penaliza eso). El lado de scikit-learn evita este problema con una sola
línea, `class_weight="balanced"`.

Con una máquina local disponible (Ryzen 5 5600G, 64GB RAM), `03`, `05` y `06`
se re-ejecutaron **localmente** — no solo para reemplazar Colab, sino para
corregir ese modelo. La corrección (pesos balanceados por clase vía
`weightCol`, calculados a mano con la misma fórmula que usa scikit-learn
internamente) y su efecto están documentados en `05_modeling_spark.ipynb` y
discutidos en las [Conclusiones](conclusions.md).

Migrar de Colab a Windows local también obligó a resolver una serie de
problemas específicos de plataforma que no existían en Colab (Linux): PySpark
necesita `winutils.exe`/`hadoop.dll` incluso para escribir Parquet en el
filesystem local de Windows; la config de memoria de Spark, calculada
dinámicamente a partir de los recursos reales de la sesión (ver
`03_preprocessing_spark.ipynb`), tenía un techo de `8g` heredado del
enunciado que resultó insuficiente en esta máquina — con 12 hilos lógicos,
`local[*]` lanza hasta 12 tareas concurrentes de entrenamiento compitiendo
por ese mismo heap, y provocó `OutOfMemoryError` (más paralelismo generó más
presión de memoria *concurrente*, no menos); y el wrapper de Python de
`BinaryClassificationMetrics.roc()` (API RDD de `pyspark.mllib`) dejó de
estar disponible en PySpark 3.5.x, por lo que la curva ROC de Spark se
recalculó con un método alternativo 100% distribuido (agregación por bins de
probabilidad). Todo esto está documentado inline en los notebooks
correspondientes.

`01` (EDA) no se volvió a ejecutar: no depende del modelo de Spark y sus
resultados de Colab siguen siendo válidos.

## Estructura del libro

1. **Preparación de datos** — descarga, escaneo completo de las 151 columnas
   originales, selección justificada de variables, limpieza.
2. **EDA** — análisis uni y bidimensional, valores faltantes, resumen ejecutivo.
3. **Preprocesamiento** — scikit-learn y PySpark, cada uno con su propio
   pipeline independiente.
4. **Modelado** — `RandomForestClassifier` con tuning de hiperparámetros en
   ambos frameworks (`GridSearchCV` vs `ParamGridBuilder`+`CrossValidator`).
5. **Interpretabilidad** — LIME sobre instancias mal clasificadas de ambos
   modelos.
6. **Comparación** — métricas, tiempos, curva ROC.
7. **Conclusiones** — reflexión crítica sobre velocidad, precisión y el efecto
   de cada condición obligatoria del proyecto.
