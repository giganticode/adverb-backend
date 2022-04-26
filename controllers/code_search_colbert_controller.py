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

        if content:
            content = json.loads(str(content))

        data = []
        for key in content:
            file_content = str(content[key]["content"])
            if file_content:
                file_content = file_content.replace("\r\n", " ").replace("\n", " ")
                data.append(file_content)
        
        collection = Collection(data=data)
        checkpoint = os.path.join(os.getcwd(), "models", "colbertv2.0")

        nbits = 2   # encode each dimension with 2 bits
        doc_maxlen = 300   # truncate passages at 300 tokens
        nranks = 1 if torch.cuda.is_available() else 0 # number of gpu's to use
        with Run().context(RunConfig(nranks=nranks)):
            config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)
            config.local_files_only = True  # use local indexer checkpoint
            indexer = Indexer(checkpoint=checkpoint, config=config)
            indexer.index(name=index_name, collection=collection, overwrite=True)
            # print(indexer.get_index())
        
        return ""


    def search_for_text(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        query = data.get("search", "")
        index_name = data.get("index_name", "")
        
        with Run().context(RunConfig()):
            searcher = Searcher(index=index_name)

        results = searcher.search(query, k=3)
        
        return_values = []
        for passage_id, passage_rank, passage_score in zip(*results):
            return_values.append({"id": passage_id, "rank": passage_rank, "score": passage_score})
        return return_values