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

for sentence in sentences:
    sentence_encoding = model.encode(sentence)

    print(float(util.pytorch_cos_sim(search_encode, sentence_encoding)))
