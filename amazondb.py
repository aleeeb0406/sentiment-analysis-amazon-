import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from wordcloud import WordCloud


nltk.download('vader_lexicon')
nltk.download('stopwords')

df = pd.read_csv("amazon.csv")


print("Vista previa del dataset:")
print(df.head())


print("\nColumnas del dataset:")
print(df.columns)


texto_columna = 'review_content'  


df = df.dropna(subset=[texto_columna])  
df[texto_columna] = df[texto_columna].astype(str)


sia = SentimentIntensityAnalyzer()

df['sentiment_score'] = df[texto_columna].apply(lambda x: sia.polarity_scores(x)['compound'])


def clasificar_sentimiento(score):
    if score >= 0.05:
        return 'Positivo'
    elif score <= -0.05:
        return 'Negativo'
    else:
        return 'Neutro'

df['sentiment'] = df['sentiment_score'].apply(clasificar_sentimiento)


print("\nDistribución de sentimientos:")
print(df['sentiment'].value_counts())


plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=df, palette='coolwarm', order=['Positivo', 'Neutro', 'Negativo'])
plt.title('Distribución de Sentimientos en Reseñas de Amazon')
plt.xlabel('Tipo de Sentimiento')
plt.ylabel('Cantidad de Reseñas')
plt.show()


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


texto_positivo = " ".join(review for review in df[df['sentiment']=='Positivo'][texto_columna])
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(texto_positivo)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras - Reseñas Positivas')
plt.show()


if 'rating' in df.columns:
    plt.figure(figsize=(6,4))
    sns.barplot(x='rating', y='sentiment_score', data=df, palette='viridis')
    plt.title('Relación entre Rating y Puntaje de Sentimiento')
    plt.xlabel('Rating del Producto')
    plt.ylabel('Puntaje Promedio de Sentimiento')
    plt.show()


df.to_csv("amazon_sentiment_analysis_results.csv", index=False)
print("\nArchivo con resultados exportado correctamente ✅")
