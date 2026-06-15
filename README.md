
#  Análisis de Sentimientos en Reseñas de Amazon

## Descripción
Análisis de sentimientos sobre reseñas de productos de Amazon utilizando 
Procesamiento de Lenguaje Natural (NLP). El proyecto clasifica reseñas como 
Positivas, Neutras o Negativas, genera visualizaciones y exporta los 
resultados para su análisis posterior.

## Tecnologías utilizadas
- Python
- Pandas
- NLTK (VADER Sentiment Analyzer)
- Seaborn
- Matplotlib
- WordCloud

## Dataset
Dataset de reseñas de Amazon obtenido de [[Kaggle](https://www.kaggle.com)](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews).  
Contiene más de 5,000 reseñas con columnas de contenido de texto y rating.

## Análisis realizados
- Clasificación de sentimientos (Positivo / Neutro / Negativo) con VADER
- Distribución de sentimientos con gráfica de barras
- Nube de palabras de reseñas positivas
- Relación entre rating del producto y puntaje de sentimiento
- Exportación de resultados a CSV

## Resultados
Cada reseña recibe un puntaje compuesto (compound score):
- **Positivo:** score ≥ 0.05
- **Neutro:** entre -0.05 y 0.05
- **Negativo:** score ≤ -0.05

## Cómo ejecutarlo

### 1. Instala las dependencias
pip install pandas seaborn matplotlib nltk wordcloud

### 2. Descarga el dataset de Kaggle y nómbralo
amazon.csv

### 3. Ejecuta el script
python amazondb.py

### 4. Resultados
Se generará el archivo amazon_sentiment_analysis_results.csv 
con la clasificación de cada reseña.

## Estructura del proyecto
├── amazondb.py                          # Script principal
├── amazon.csv                           # Dataset (no incluido)
└── amazon_sentiment_analysis_results.csv # Resultados generados
