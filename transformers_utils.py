from transformers import AutoTokenizer, AutoModel, pipeline

def create_autotokenizer(model="bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model)

def create_automodel(model="bert-base-uncased"):
    return AutoModel.from_pretrained(model)
