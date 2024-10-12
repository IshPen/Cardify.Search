from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from transformers import BertTokenizer, BertModel, pipeline
import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


# Preprocessing function
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    return tokens


# Embedding function
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings


# Sentiment analysis function
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')


def get_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label']


# KNN search function
def knn_search(search_phrase, database_phrases, k=3):
    search_embedding = get_embedding(search_phrase)
    database_embeddings = [get_embedding(phrase) for phrase in database_phrases]

    distances = euclidean_distances(search_embedding, database_embeddings)[0]
    nearest_indices = np.argsort(distances)[:k]

    nearest_phrases = [database_phrases[i] for i in nearest_indices]
    nearest_distances = [distances[i] for i in nearest_indices]

    return nearest_phrases, nearest_distances


# Ranking and selection function
def find_best_match(search_phrase, database_phrases, k=3):
    search_sentiment = get_sentiment(search_phrase)
    print(f"Search Phrase Sentiment: {search_sentiment}")

    nearest_phrases, nearest_distances = knn_search(search_phrase, database_phrases, k)

    best_match = None
    for phrase, distance in zip(nearest_phrases, nearest_distances):
        phrase_sentiment = get_sentiment(phrase)
        print(f"Phrase: {phrase}, Distance: {distance}, Sentiment: {phrase_sentiment}")
        if phrase_sentiment == search_sentiment:
            best_match = phrase
            break

    return best_match


# Example usage
search_phrase = "licensing good"
database_phrases = [
    "Without licensing, AI erodes trust in information, causes newspaper closure, and leads to mass job loss of journalists—that kills democracy.",
    "AI erodes search traffic for journalism",
    "Licensing helps maintain the quality and trustworthiness of information."
]

best_match = find_best_match(search_phrase, database_phrases, k=3)
print(f"Best Match: {best_match}")
