from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

search = "Licensing Important"
sentences = [
    'Lawsuits don’t solve—they’ll rule in favor of AI, they’re case-by-case, and legislation is key',
    'Without licensing, AI erodes trust in information, causes newspaper closure, and leads to mass job loss of journalists—that kills democracy.',
    'AI-driven erosion of trust in democracy in the US causes global rise in autocracy and war',
    "AI erodes search traffic for journalism"
]
search_encode = model.encode(search)

sentence = sentences[1]

sum = 0
for term1 in search.split():
    for term2 in sentence.split():

        print(f'{term1} {term2}')
        cosine = util.pytorch_cos_sim(model.encode(term1), model.encode(term2))
        print(cosine)
        sum+=float(cosine)
print(sum)
print(sum/(len(search.split())*len(sentence.split())))