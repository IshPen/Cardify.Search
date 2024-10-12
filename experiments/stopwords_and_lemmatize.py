import time

from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import time
stop_words = set(stopwords.words('english'))
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
lemmatizer = WordNetLemmatizer()

search = "Licensing Important"
sentences = [
    'Lawsuits don’t solve—they’ll rule in favor of AI, they’re case-by-case, and legislation is key',
    'Without licensing, AI erodes trust in information, causes newspaper closure, and leads to mass job loss of journalists—that kills democracy.',
    'AI-driven erosion of trust in democracy in the US causes global rise in autocracy and war',
    "AI erodes search traffic for journalism"
    ]
def get_pos(search):
    search_tokenized = sent_tokenize(search)
    search_to_list = nltk.word_tokenize(search_tokenized[0])
    start = time.time()
    search_to_list = [lemmatizer.lemmatize(w, pos="a") for w in search_to_list if not w in stop_words]
    print("Time:", time.time()-start)
    search_tagged = nltk.pos_tag(search_to_list)
    lemmatized_reconstructed_sentence = ' '.join([word for word, tag in search_tagged])
    print("L:", lemmatized_reconstructed_sentence)
    return search_tagged

sentences = [sentence for sentence in sentences]
search_encode = model.encode(search)
search_pos = get_pos(search)


print(search_pos)


for sentence in sentences:
    part_of_speech = get_pos(sentence)
    print(part_of_speech)
    sentence_encoding = model.encode(sentence)

    print(float(util.pytorch_cos_sim(search_encode, sentence_encoding)))

