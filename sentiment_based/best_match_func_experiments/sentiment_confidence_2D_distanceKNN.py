import os
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    return tokens

# Embedding function
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings


# Save embeddings to a file
def save_embeddings(database_phrases, filename='embeddings.pkl'):
    embeddings = {phrase: get_embedding(phrase) for phrase in database_phrases}
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)


# Load embeddings from a file
def load_embeddings(filename='embeddings.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings
    else:
        return None