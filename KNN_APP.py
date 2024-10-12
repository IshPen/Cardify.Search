# CELL 1
import os
import pickle
import warnings
warnings.filterwarnings("ignore", message="A parameter name that contains `beta` will be renamed internally to `bias`. Please use a different name to suppress this warning.")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import plotly.graph_objects as go
import streamlit as st
from st_copy_to_clipboard import st_copy_to_clipboard
from transformers_utils import create_autotokenizer, create_automodel


if "CAN_SEARCH" not in st.session_state:
    st.session_state["CAN_SEARCH"] = True
if "NEEDS_TO_LOAD_EMBEDDINGS" not in st.session_state:
    st.session_state["NEEDS_TO_LOAD_EMBEDDINGS"] = True
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = None
if "has-pre-run" not in st.session_state:
    st.session_state["has-pre-run"] = False

#if st.session_state["has-pre-run"] == False:
print("Initializing System")
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

nltk.download('wordnet')

# CELL 3
# Embedding function
tokenizer = create_autotokenizer('bert-base-uncased')
model = create_automodel('bert-base-uncased')

st.session_state["has-pre-run"] = True
print("System Init Finished")

# CELL 2
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    return tokens



# CELL 4
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
        # print("embeddings", embeddings)
        st.session_state["embeddings"] = embeddings
        return embeddings
    else:
        # print("no embeddings")
        return None

# CELL 5
# Semantic similarity with loaded embeddings
def semantic_similarity(search_phrase, database_phrases, embeddings):
    search_embedding = get_embedding(search_phrase)
    if embeddings is None:
        print("ss embeddings = none")
    print(f'Embeddings Length: {len(embeddings)}')
    print(type(embeddings))
    print(f'E Keys: {embeddings.keys()}')
    similarities = [cosine_similarity(search_embedding, embeddings[phrase])[0][0] for phrase in database_phrases]
    return similarities

def normalize_term(phrase):
    return phrase.replace(" ", "")

def calc_literal_distance(lemmatized_search, search, phrase):
    # split_search = re.split(r"; <>:;!@#%&", search).split(" ")
    split_search = wordpunct_tokenize(search)
    normalized_phrase = normalize_term(phrase=phrase)

    dist = 0
    for lemmed_term in lemmatized_search:
        # print(lemmed_term)
        if lemmed_term in normalized_phrase:
            dist += len(lemmed_term)**1.3
            print(lemmed_term)

    print(split_search)
    print(normalized_phrase)
    return dist, normalized_phrase

# CELL 6
def eucl_distance_func(x1, x2, y1, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# CELL 7
def find_best_match_with_lemming_knn(search_phrase, database_phrases, embeddings, n_neighbors=3):
    # Get sentiment of search
    # search_sentiment, search_confidence = get_sentiment(search_phrase)
    # print(f"Search Phrase Sentiment: {search_sentiment} (Confidence: {search_confidence:.4f})")

    # Get Semantic Similarities
    cosine_similarities = semantic_similarity(search_phrase, database_phrases, embeddings)

    # Lemmatize the search term
    split_search = wordpunct_tokenize(search_phrase)
    lemmatized_split_search = [stemmer.stem(lemmatizer.lemmatize(search)) for search in split_search]

    normalized_phrases = []
    literal_distances = []
    for phrase in database_phrases:
        distance, normalized_phrase = calc_literal_distance(lemmatized_search=lemmatized_split_search,
                                                            search=search_phrase, phrase=phrase)
        print(f"Score: {round(distance, 4)}: {normalized_phrase}")
        literal_distances.append(distance)
        normalized_phrases.append(normalized_phrase)

    print(f'Literal Distances: {literal_distances}')
    print(f'Normalized Phrases: {normalized_phrases}')

    ## Sort Literal Distances and print
    literally_ranked_phrases = sorted(zip(database_phrases, literal_distances), key=lambda x: x[1], reverse=True)
    print(f'Literally Ranked Phrases: {literally_ranked_phrases}')

    # Zip Phrases with Cosine Similarities
    # Sort by most to least cosine similar
    cosine_ranked_phrases = sorted(zip(database_phrases, cosine_similarities), key=lambda x: x[1], reverse=True)
    print(f'Cosine Ranked Phrases: {cosine_ranked_phrases}')

    # Zip Literal Distances and Cosine Similarities
    ranked_phrases = sorted(zip(database_phrases, literal_distances, cosine_similarities), key=lambda x: x[1],
                            reverse=True)
    print(f'Ranked Phrases: {ranked_phrases}')

    # KNN Calculations
    distances = []
    distance_sorted_phrases = []

    literal_dist_uniform_scaler = 1 / max(literal_distances) if max(literal_distances) != 0 else 1

    for phrase, literal_distance, cosine_similarity in ranked_phrases:
        distance = eucl_distance_func(max(literal_distances) * literal_dist_uniform_scaler,
                                      literal_distance * literal_dist_uniform_scaler, 1, cosine_similarity)
        print(distance, phrase)
        distances.append(distance)
        distance_sorted_phrases.append(phrase)

    zipped_knn_phrase_distance = sorted(zip(distance_sorted_phrases, distances), key=lambda x: x[1])
    print(f'Zipped KNN Phrase Distance: {zipped_knn_phrase_distance}')
    # print(distances)
    # print(distance_sorted_phrases)

    n_neighbors_to_return = []
    for i in range(n_neighbors):
        n_neighbors_to_return.append(zipped_knn_phrase_distance[i])
    print(n_neighbors_to_return)

    ### ---------- PLOTTING ---------- ###

    # Prepare data for the plot
    x_values = cosine_similarities + [1]  # Cosine similarities and search phrase at x=1
    y_values = []
    hover_texts = []
    for phrase, cosine_similarity, literal_distance in zip(database_phrases, cosine_similarities, literal_distances):
        # phrase_sentiment, phrase_confidence = get_sentiment(phrase)
        # y_value = phrase_confidence if phrase_sentiment == 'POSITIVE' else -(phrase_confidence)
        y_value = literal_distance
        y_values.append(y_value * literal_dist_uniform_scaler)
        hover_texts.append(
            f"Phrase: {phrase}<br>Similarity: {cosine_similarity:.4f}<br>Literal Distance: {literal_distance * literal_dist_uniform_scaler}")

    # Add search phrase to the plot
    y_search = max(literal_distances) * literal_dist_uniform_scaler
    y_values.append(y_search)
    hover_texts.append(
        f"Search Phrase: {search_phrase}<br>Similarity: {int(1):.4f} (Literal Distance: {max(literal_distances) * literal_dist_uniform_scaler:.4f})")

    # Create the scatter plot
    fig = go.Figure(data=[go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers',
        marker=dict(size=10, color=y_values, colorscale='Viridis', showscale=True),
        text=hover_texts,
        hoverinfo='text'
    )])

    # Set axis labels and title
    fig.update_layout(
        xaxis_title="Cosine Similarity",
        yaxis_title="Sentiment Confidence",
        title="Semantic Similarity and Sentiment",
        xaxis_range=[-0.1, 1.1],  # Adjust range to accommodate labels
        yaxis_range=[-1.1, 1.1]
    )

    for phrase, distance in n_neighbors_to_return:
        st.text(phrase)
        # if st.button(phrase):
        # st_copy_to_clipboard(phrase)
        #st_copy_to_clipboard(
        #    text=str(phrase),
        #    before_copy_label=str(phrase),
        #    after_copy_label=str(phrase)+"✅",
        #    key=str(phrase)
        #    )

    # Display the plot
    # fig.show()
    st.plotly_chart(fig, use_container_width=True)
    ### ^^^^^^^^^^ PLOTTING ^^^^^^^^^^ ###

    return n_neighbors_to_return

database_phrases = ['Notes', '1ac—plan', '1ac—journalism advantage', '1ac—model collapse', '1ac—facial recognition',
 '2ac at: licensing fails—top level', '2ac/1ar at: licensing fails—at: can’t administer',
 '2ac/1ar at: licensing fails—at: scale', '2ac/1ar at: licensing fails—at: complexity',
 '2ac/1ar at: licensing fails—at: no market', '2ac/1ar at: licensing fails—at: only helps large publishers',
 '2ac/1ar at: licensing fails—at: Damle', '2ac at: absolution turn', '2ac at: consolidation alt cause',
 '2ac at: no democracy impact', '2ac at: no misinfo impact', '2ac—at: squo solves licensing',
 '2ac—at: lawsuits solve', 'uq—ai hurts journalism', 'i/l—licensing solves journalism',
 'i/l—broadcasting solves democ/misinfo', 'i/l—at: ai solves journalism', 'i/l—journalism k2 democracy',
 'i/l—journalism solves misinformation', 's—US key', 's—at: fair use', '2ac at: no model collapse',
 '1ar at: no model collapse', '2ac at: licensing doesn’t solve', '2ac at: human content doesn’t solve',
 '2ac at: model collapse doesn’t cause poisoning', '2ac at: no data poisoning impact ', '2ac case turns ai da',
 '2ac at: can’t solve FRT', '2ac at: FRT not key to LAWS', '2ac at: no LAWS impact',
 '2ac at: can’t solve deep fakes', '2ac at: no deep fake impact', '2ac !—carcerality', '2ac non-unique',
 '1ar non-unique', '2ac no link', '1ar no link—wall ', '1ar at: link/solvency doublebind', '1ar at: cost link',
 '1ar at: move abroad link', '1ar no link—indict of neg', '1ar no link—at: open ai', '2ac link turn',
 '1ar link turn', '1ar link turn—competitiveness link', '2ac creativity turn', '1ar creativity turn—licensing key',
 '1ar creativity turn—humans key', '2ac no ai impact', '1ar no ai impact', '2ac/1ar Case outweighs', '***Note',
 '2ac military turn', '1ar military turn—innovation = military ai', '1ar military turn—military ai = war',
 '1ar military turn—at: humans detect failures', '2ac superintelligence turn',
 '1ar superintelligence turn —yes extinction', '1ar superintelligence turn —at: AI is nice',
 '1ar superintelligence turn —pause key', '1ar superintelligence turn —at: agi impossible',
 '1ar superintelligence turn —at: agi too far off', '2ac non-unique', '2ac commercial only', '2ac licensing solves',
 '1ar licensing solves', '2ac fair use doesn’t solve', '2ac research not key', '1ar research not key',
 '2ac no impact', '1ar no impact', '2ac link turn', '1ar link turn', '1ar uq—copyright clog now',
 '2ac certainty deficit', '2ac flexibility deficit', '2ac at: licensing pic', '2ac opt out fails',
 '1ar at: sufficiency framing', '1ar opt out fails—tech ', '1ar opt out fails—search traffic',
 '1ar opt out fails—onus', '1ar at: robots.txt', '2ac delay deficit', '1ar delay deficit', '2ac certainty deficit',
 '1ar congress key—certainty', 'preemption deficit', '2ac at: compulsory licensing cp',
 '1ar at: compulsory license cp']

# Load embeddings if they exist, otherwise compute and save them
embeddings = load_embeddings()

# embeddings = None
# If embeddings.pkl doens't exist OR the database_phrases doesn't have the same indexes as embeddings (ignoring duplicates)

if embeddings is None or len(list(set(database_phrases))) != len(list(set(embeddings))):
    print("embeddings = none")
    print(f'EmbeddingsLen: {len(embeddings)}, DatabasePhraseLen: {len(database_phrases)}')
    print(f'Embeddings: {embeddings.keys()}'
          f'DatabasePhrases: {database_phrases}')
    # st.cache_data.clear()
    save_embeddings(database_phrases)
    embeddings = load_embeddings()


search_phrase = "2ac compulsory licensing"

# Custom KNN Literal Distance COSINE MAXING
if st.session_state["CAN_SEARCH"] == True:
    top_matches = find_best_match_with_lemming_knn(search_phrase, database_phrases, embeddings, n_neighbors=5)
    print(f"Dist {top_matches}")
    st.session_state["CAN_SEARCH"] = False


search_query = st.text_input("Search Document")
if search_query != "":
    print("not empty")

    st.session_state["CAN_SEARCH"] = True
    top_matches = find_best_match_with_lemming_knn(search_query, database_phrases, embeddings, n_neighbors=5)
    print(f"Dist {top_matches}")
    st.session_state["CAN_SEARCH"] = False

