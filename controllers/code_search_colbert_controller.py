import os 
from flask import json
from flask.wrappers import Request
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
import torch

class CodeSearchColBertController:

    def indexing(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        content = data.get("content", "")
        index_name = data.get("index_name", "")

        print("step")
        data = list(map(lambda x: x["content"], list(content.values())))
        collection = Collection(data=content)

        # collection = Collection(path="C:\\adverb-backend\\controllers\\downloads\\lotte\\science\\dev\\collection - Kopie.tsv")

        checkpoint = os.path.join("..", "models", "colbertv2.0")
        print(checkpoint)
        nbits = 2   # encode each dimension with 2 bits
        doc_maxlen = 300   # truncate passages at 300 tokens
        nranks = 1 if torch.cuda.is_available() else 0 # number of gpu's to use
        with Run().context(RunConfig(nranks=nranks)):
            config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)
            print("step 1")
            indexer = Indexer(checkpoint=checkpoint, config=config)
            print("step 2")
            indexer.index(name=index_name, collection=collection, overwrite=True)
            print("step 3")
            print(indexer.get_index())
        
        return ""


    def search_for_text(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        query = data.get("search", "")
        index_name = data.get("index_name", "")
        
        # with Run().context(RunConfig(experiment='notebook')):
        #     searcher = Searcher(index=index_name)

        # results = searcher.search(query, k=3)
        
        return ""