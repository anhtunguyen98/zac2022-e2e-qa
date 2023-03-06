"""
    Encode all wiki passage to 768 embedding
"""
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os

model_name = 'checkpoint_18_11'
model = SentenceTransformer(model_name, device='cuda')

corpus_path = "sub_wiki_len30_cleaned.json"

corpus_embeddings_save_path = "wiki_passage_embeddings.npy"

# Check if embedding cache path exists
if not os.path.exists(corpus_embeddings_save_path):
    # Get all unique sentences from the file
    corpus_passages = []
    with open(corpus_path, encoding='utf8') as fIn:
        corpus = json.load(fIn)
    for passage in corpus:
        title = passage["title"]
        text = passage['text']
        if title != "":
          text = title + ' </s> ' + text
        corpus_passages.append(text)
    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_passages, show_progress_bar=True, convert_to_numpy=True, batch_size=1000)

    print("Store file on disc")
    with open(corpus_embeddings_save_path, "wb") as fOut:
        np.save(corpus_embeddings, fOut)
