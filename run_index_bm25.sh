python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input  bm25_corpus\
  --index sparse_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4