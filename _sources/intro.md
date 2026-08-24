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
