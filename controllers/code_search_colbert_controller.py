from utils.logging import log
import os 
from flask import json
from flask.wrappers import Request
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Collection
from colbert import Indexer, Searcher
import torch

class CodeSearchColBertController:

    def indexing(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        content = data.get("content", "")
        index_name = data.get("index_name", "adverb")
        if not content:
            return None
       
        collection = self.convert_json_to_collection(content)
        checkpoint = os.path.join(os.getcwd(), "models", "colbertv2.0")

        nbits = 2   # encode each dimension with 2 bits
        doc_maxlen = 300   # truncate passages at 300 tokens
        nranks = 1 if torch.cuda.is_available() else 0 # number of gpu's to use
        with Run().context(RunConfig(nranks=nranks)):
            config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)
            config.local_files_only = True  # use local indexer checkpoint
            config.overwrite = True
            indexer = Indexer(checkpoint=checkpoint, config=config, )
            indexer.index(name=index_name, collection=collection, overwrite=True)
            # logging.log(indexer.get_index())
        
        return ""

    def search_for_text(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        query = data.get("search", "")
        content = data.get("content", "")
        index_name = data.get("index_name", "adverb")
        if not query or not content:
            return None

        collection = self.convert_json_to_collection(content)
        checkpoint = os.path.join(os.getcwd(), "models", "colbertv2.0")
        
        with Run().context(RunConfig()):
            searcher = Searcher(index=index_name, checkpoint=checkpoint, collection=collection)

        results = searcher.search(query, k=5)
        
        return_values = []
        for passage_id, passage_rank, passage_score in zip(*results):
            return_values.append({"index": passage_id, "match": [0], "rank": passage_rank, "score": passage_score})

        log("Search NL->PL - model:", "colbert")
        log("Search NL->PL - query:", query)
        log("Search NL->PL - result:", str(return_values))
        
        return return_values

    def convert_json_to_collection(self, content):
        content = json.loads(str(content))
        data = []
        for item in content:
            file_content = str(item["content"])
            if file_content:
                file_content = file_content.replace("\r\n", " ").replace("\n", " ")
                data.append(file_content)

        collection = Collection(data=data)
        return collection