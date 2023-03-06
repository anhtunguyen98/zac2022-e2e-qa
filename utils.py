import os
import string
import json
from typing import List
from pyvi import ViTokenizer
import unicodedata as ud
import logging




vi_stop_word = ["như", "làm", "là", "và", "với", "nếu", "thì", "do", "ở", "đây", "đó", "lại", "không", "nhỉ", "ta",
                "cho", "chung", "đã", "nơi", "để", "đến", "số", "một", "khác", "được", "vào", "ra", "trong", "ạ",
                "người", "loài", "từ", "nào", "bằng", "rằng", "nên", "gì", "việc", "ấy", "khi", "này", "chỉ", "về",
                "các", "còn", "trên", "những", "có", "mà", "nhưng", "nhiều", "nó", "sẽ", "chưa", "lúc", "có_thể",
                "bởi_vì", "tại_vì", "như_thế", "thế_là", "trong_khi", "thế_mà", "chẳng_hạn", "do_đó", "tuy_nhiên",
                "đôi_khi", "chỉ_là", "một_số", "chúng_nó", "rằng_là", "thứ", "của"]

number = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

chars = ["a", "b", "c", "d", "đ", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"]


def load_corpus_refer(path):
    return json.load(open(path, 'r'))


def word_tokenize(text: str):
    text_tokens = ViTokenizer.tokenize(text)
    return text_tokens


def remove_punctuation(w):
    return w not in string.punctuation


def lower_case(w):
    return w.lower()


def remove_stop_word(w):
    return w not in vi_stop_word + number + chars


def bm25_tokenizer(tokens: List[str]):
    """Pre-processing input for bm25 search"""
    tokens = list(map(lower_case, tokens))
    tokens = list(filter(remove_punctuation, tokens))
    tokens = list(filter(remove_stop_word, tokens))
    return " ".join(tokens)


def model_tokenizer(tokens: List[str], low_case: bool = False, remove_punc: bool = False):
    """Pre-processing input for dense search"""
    if low_case is True:
        tokens = list(map(lower_case, tokens))
    if remove_punc is True:
        tokens = list(filter(remove_punctuation, tokens))
    return " ".join(tokens)


def normalize(text: str):
    """Normalize passage text"""
    text = ud.normalize("NFC", text)
    text = " ".join(text.split())
    text = text.replace("–", "")
    text = text.replace("‘", "'")
    text = text.replace('"', '')
    text = text.replace("'", "")
    text = text.replace("”", "")
    text = text.replace("“", "")
    text = text.replace("′", "")
    text = text.replace("...", "")
    # text = "".join([char for char in text if ord(char) < 8000])
    return text.strip()


def hybird_search(sparse_results, dense_results, top_k=100):
    hybrid_results = []
    for sparse_hits, dense_hits in zip(sparse_results, dense_results):
        hybrid_result = {}
        min_dense_score = min(dense_hits.values()) if len(dense_hits) > 0 else 0
        # max_dense_score = max(dense_hits.values()) if len(dense_hits) > 0 else 1
        min_sparse_score = min(sparse_hits.values()) if len(sparse_hits) > 0 else 0
        # max_sparse_score = max(sparse_hits.values()) if len(sparse_hits) > 0 else 1
        for doc in set(dense_hits.keys()) | set(sparse_hits.keys()):
            if doc not in dense_hits:
                sparse_score = sparse_hits[doc]
                dense_score = min_dense_score
            elif doc not in sparse_hits:
                sparse_score = min_sparse_score
                dense_score = dense_hits[doc]
            else:
                sparse_score = sparse_hits[doc]
                dense_score = dense_hits[doc]
            score = sparse_score * dense_score
            hybrid_result[doc] = score
        hybrid_results.append(sorted(hybrid_result.items(), key=lambda x: x[1], reverse=True)[:top_k])
    return hybrid_results
