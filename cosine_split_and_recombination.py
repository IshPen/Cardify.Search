from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, util
import re
from scipy.spatial import distance
import nltk
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')


# model = SentenceTransformer('distilbert-base-nli-mean-tokens')
model = SentenceTransformer('all-MiniLM-L6-v2')

search = "AI Causes Misinformation"
sentences = [
    'Lawsuits don’t solve—they’ll rule in favor of AI, they’re case-by-case, and legislation is key',
    'Without licensing, AI erodes trust in information, causes newspaper closure, and leads to mass job loss of journalists—that kills democracy.',
    'AI-driven erosion of trust in democracy in the US causes global rise in autocracy and war',
    "AI erodes search traffic for journalism"
    ]

search_list = search.split()
print(search_list)
docs_split = sentences[2].split()
print(docs_split)

def individual_cosine_similarity(term1, term2, encoding_model=model, exponential_weight=12):
    # Normalize First
    term1 = term1.lower()
    term2 = term2.lower()
    term1 = re.sub("[!@#$%^&*()<>?,./;':{}-]", "", term1)
    term2 = re.sub("[!@#$%^&*()<>?,./;':{}-]", "", term2)
    print(term1)
    print(term2)

    # Encode
    term1encoded = encoding_model.encode(term1)
    term2encoded = encoding_model.encode(term2)

    # 1v1 cosine similarity
    return float(util.pytorch_cos_sim(term1encoded, term2encoded))**exponential_weight

def multidimensional_euclidean_distancing(term1, term2, encoding_model=model, exponential_weight=12):
    # Normalize First
    term1 = term1.lower()
    term2 = term2.lower()
    term1 = re.sub("[!@#$%^&*()<>?,./;':{}-]", "", term1)
    term2 = re.sub("[!@#$%^&*()<>?,./;':{}-]", "", term2)
    #print(term1)
    #print(term2)

    # Encode
    term1encoded = encoding_model.encode(term1)
    term2encoded = encoding_model.encode(term2)
    # print(term1encoded)

    dist = distance.euclidean(term1encoded, term2encoded)
    #print(dist)
    # 1v1 cosine similarity
    return float(util.pytorch_cos_sim(term1encoded, term2encoded))**exponential_weight

def custom_cosine_similarity_maxing(term1, term2, encoding_model=model, exponential_weight=3):
    # Normalize First
    term1 = term1.lower()
    term2 = term2.lower()
    term1 = re.sub("[!@#$%^&*()<>?,./;':{}-]", "", term1)
    term2 = re.sub("[!@#$%^&*()<>?,./;':{}-]", "", term2)
    #print(term1)
    #print(term2)

    # Encode
    term1encoded = encoding_model.encode(term1)
    term2encoded = encoding_model.encode(term2)

    # 1v1 cosine similarity
    cos_similarity = float(util.pytorch_cos_sim(term1encoded, term2encoded))**exponential_weight
    return min(cos_similarity*2, 1)

term1 = search_list[2]

for i in range(0, len(docs_split)):
    term2 = docs_split[i]
    cos_sim = custom_cosine_similarity_maxing(term1, term2, encoding_model=model, exponential_weight=1)
    multidimensional_euclidean_distancing(term1, term2, exponential_weight=1)
    print(nltk.pos_tag(nltk.word_tokenize(term2)))
    print(f'For {term1} and {term2}: {cos_sim}')