import re
from nltk import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import time


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

start_time = time.time()

search = "wall no link"
phrase = "2ac at: no deep fake impact"
phrase = "1ar no link—wall "
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

def normalize_term(phrase):
    return phrase.replace(" ", "")

def calc_literal_distance(lemmatized_search, search=search, phrase=phrase):
    # split_search = re.split(r"; <>:;!@#%&", search).split(" ")
    split_search = wordpunct_tokenize(search)
    normalized_phrase = normalize_term(phrase=phrase)

    dist = 0
    for lemmed_term in lemmatized_search:
        # print(lemmed_term)
        if lemmed_term in normalized_phrase or lemmed_term in phrase:
            dist += len(lemmed_term)**1.3
            print(lemmed_term)

    print(split_search)
    print(normalized_phrase)
    return dist, normalized_phrase

split_search = wordpunct_tokenize(search)
lemmatized_split_search = [stemmer.stem(lemmatizer.lemmatize(search)) for search in split_search]
print(f'lem {lemmatized_split_search}')
distances = []
normalized_phrases = []

for phrase in database_phrases:
    distance, normalized_phrase = calc_literal_distance(lemmatized_search=lemmatized_split_search, search=search, phrase=phrase)
    print(f"Score: {round(distance, 4)}: {normalized_phrase}")
    distances.append(distance)
    normalized_phrases.append(normalized_phrase)

zipped_distances_list = list(zip(distances, normalized_phrases))

# Printing zipped list
print("Initial zipped list - ", str(zipped_distances_list))

# Using sorted and lambda
res = list(reversed(sorted(zipped_distances_list, key=lambda x: x[0])))

# printing result
print("final list - ", str(res))

print(time.time()-start_time)