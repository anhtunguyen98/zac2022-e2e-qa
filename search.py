import logging
import faiss
import numpy as np
import coloredlogs
import time
from utils import word_tokenize, bm25_tokenizer, model_tokenizer, hybird_search, load_corpus_refer
from pyserini.search.lucene import LuceneSearcher
from typing import List, Tuple
from cross_utils import cross_model_infer
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import pipeline
from post_process import process_answer
import json
from config import CFG
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

coloredlogs.install(
    level="INFO", fmt="%(asctime)s %(name)s[%(process)d] %(levelname)-8s %(message)s"
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Inference:

    def __init__(self, CFG) :
        self.CFG = CFG
        # Load model. index, corpus
        load_start_time = time.time()
        logger.info("Loading Sparse Searcher...!")
        self.SPARSE_SEARCHER = LuceneSearcher(CFG.INDEXES_BM25_PATH)
        self.SPARSE_SEARCHER.set_bm25(k1=CFG.BM25_K1, b=CFG.BM25_B)

        logger.info("Loading Dense Searcher...!")
        cpu_index = faiss.read_index(CFG.INDEXES_DENSE_PATH)
        res = faiss.StandardGpuResources()
        self.DENSE_SEARCHER = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        del cpu_index
        self.DUAL_MODEL_ENCODER = SentenceTransformer(CFG.DUAL_MODEL_PATH, device="cuda")

        logger.info("Loading Ranker...!")
        self.MODEL_CROSS_ENCODER = CrossEncoder(CFG.CROSS_MODEL_PATH, device="cuda")

        logger.info("Loading Corpus...!")
        self.CORPUS = load_corpus_refer(CFG.CORPUS_REFER_PATH)
        self.CORPUS_CROSS = load_corpus_refer(CFG.CORPUS_CROSS_PATH)


        load_end_time = time.time()
        logger.info(f"Loading took: {load_end_time - load_start_time}s !!!")

        self.MRC_MODEL = pipeline('question-answering', model=CFG.MRC_PATH, device=0)

    
    def search(self,query, top_k):
        start = time.time()
        query_tokens = word_tokenize(query)
        
        print("query_tokens: ", query_tokens)
        
        print("BM25 query: ", bm25_tokenizer(query_tokens.split()))
        # Lexical Search
        lexical_start = time.time()
        bm25_hits = self.SPARSE_SEARCHER.search(bm25_tokenizer(query_tokens.split()), k=self.CFG.TOP_K_RETREIVAL)
        logger.info(f"Time of sparse search: {time.time() - lexical_start}s")
        sparse_results = [{hit.docid: hit.score for hit in bm25_hits}]
        print(f"Sparse_Result: {self.CORPUS[int(list(sparse_results[0].keys())[0])]}")
        # print(query)
        # print(f"Sparse DocID: {[k for k in sparse_results[0].keys()]}")
        # print(f"Sparse Score: {[v for k, v in sparse_results[0].items()]}")
        # Semantic Search
        query_tokenized = model_tokenizer(query_tokens.split())
        
        print("dense query:", query_tokenized)
        dense_start = time.time()
        question_embedding = self.DUAL_MODEL_ENCODER.encode(query_tokenized)
        # FAISS works with inner product (dot product). When we normalize vectors to unit length, inner product is equal
        # to cosine similarity
        question_embedding = question_embedding / np.linalg.norm(question_embedding)
        question_embedding = np.expand_dims(question_embedding, axis=0)
        scores, corpus_ids = self.DENSE_SEARCHER.search(question_embedding, k=self.CFG.TOP_K_RETREIVAL)
        logger.info(f"Time of dense search: {time.time() - dense_start}s")
        dense_results = [{str(corpus_id): score for corpus_id, score in zip(corpus_ids[0], scores[0])}]

        hybrid_results = hybird_search(sparse_results, dense_results, top_k)

        doc_results = [(int(res[0]), self.CORPUS_CROSS[int(res[0])], str(res[1])) for res in hybrid_results[0]]
        start_ranking = time.time()
        ranking_results = self.ranking(query_tokenized, doc_results, top_k)
        logger.info(f"Time of ranking: {time.time() - start_ranking}s")
        final_results = {"question": query, "candidates": []}
        for rank_res in ranking_results:
            psg_id = rank_res['psg_id']
            passage = self.CORPUS[int(psg_id)]
            score_sparse = '0'
            score_dense = '0'
            if psg_id in sparse_results[0]:
                score_sparse = str(sparse_results[0][psg_id])
            if psg_id in dense_results[0]:
                score_dense = str(dense_results[0][psg_id])
            final_results['candidates'].append({'title': passage['title'], 
                                                'text': passage['text'], 
                                                'score_sparse': score_sparse, 
                                                'score_dense': score_dense,
                                                'score_hybrid': rank_res['score_hybrid'],
                                                'score_ranking': rank_res['score_ranking']
                                                })
        logger.info(f"Total time: {time.time() - start}s")
        return final_results


    def ranking(self, query: str, candidates: List[Tuple[int, str, str]], top_k=20) -> List[dict]:
        output = []
        model_inputs = [[query, cand[1]] for cand in candidates]
        scores = cross_model_infer(self.MODEL_CROSS_ENCODER, model_inputs)
        scores_arg_sort = np.argsort(-scores)
        for index in scores_arg_sort[:top_k]:
            psg_id = str(candidates[index][0])
            score_hybrid = candidates[index][2]
            output.append({'psg_id': psg_id, 'score_ranking': str(scores[index]), 'score_hybrid': score_hybrid})
        return output

    def answering(self,final_results):
        
        question = final_results['question']
        score = 0
        for cadidate in final_results['candidates'][:10]:
            context = word_tokenize(cadidate['text'].strip()).replace('_', ' ')
            QA_input = {
            'question': question,
            'context': context
            }
            res = self.MRC_MODEL(QA_input)

            if score < res['score']  * float(cadidate['score_hybrid']) * float(cadidate['score_ranking']):
                title = cadidate['title']
                answer = res['answer']
                score = res['score'] * float(cadidate['score_hybrid']) * float(cadidate['score_ranking'])
                


        return answer, title


# with open('data/zac2022_testb_only_question.json','r') as f:
#     samples = json.load(f)['data']

# with open(CFG.TITLE_PATH, 'r') as f:
#     all_titles = f.read().splitlines()
# all_titles = {key.lower(): key for key in all_titles}

# with open(CFG.ENTITIES_PATH,'r') as f:
#     entities = json.load(f)
# inference = Inference(CFG)

# results = []

# for sample in samples[:]:

#     question = sample['question']
#     id = sample['id']
    
#     final_results = inference.search(question, top_k=20)
#     #print( final_results)
#     answer, title = inference.answering(final_results)
#     sample['answer'] = answer
#     sample['title'] = title
#     result = process_answer(sample,entities, all_titles)
#     results.append(result)

# submission = {'data': results}
# with open('submission.json','w') as f:
#     json.dump(results,f,ensure_ascii=False,indent=4)

