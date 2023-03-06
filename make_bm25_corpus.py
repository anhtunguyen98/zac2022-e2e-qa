import json
from utils import normalize, bm25_tokenizer
from tqdm import tqdm


def clean_text(text):
    return normalize(bm25_tokenizer(text.split()))

print("Loading corpus !!!")
with open("corpus/wiki_corpus_cleaned.json", "r", encoding='utf-8') as f:
    data = json.load(f)

print("Cleaning corpus !!!")
idx = 0
corpus_bm25 = []

for passage in tqdm(data):
    title = passage['title']
    context = passage['text']
    text = title + " " + context
    context_cleaned = clean_text(text)
    corpus_bm25.append({"id": idx, "contents": context_cleaned})
    idx += 1

print("Saving corpus !!!")
with open("bm25_corpus/corpus_pyserini.jsonl", "w") as f:
    for passage in corpus_bm25:
        json.dump(passage, f, ensure_ascii=False)
        f.write("\n")