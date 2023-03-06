
class CFG:
    # Config
    TOP_K_RETREIVAL = 200 # number of candidates return by retrieval module (bm25 x dense)
    TOP_K_RANKING = 50 # number of candidates return by cross ranker after do ranking
    BM25_K1 = 0.5
    BM25_B = 0.5
    INDEXES_BM25_PATH = "weights/sparse_index"
    INDEXES_DENSE_PATH = "weights/dense_index/index"
    DUAL_MODEL_PATH = "weights/dual_checkpoint"
    CROSS_MODEL_PATH = "weights/cross_checkpoint"
    CORPUS_REFER_PATH = "data/corpus/wiki_corpus.json"
    CORPUS_CROSS_PATH = "data/corpus/wiki_corpus_cleaned.json"
    TITLE_PATH = 'data/wikipedia_20220620_all_titles.txt'
    MRC_PATH = 'weights/xlm-mrc-large'
    ENTITIES_PATH = 'data/entities.json'