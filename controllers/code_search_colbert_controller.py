from flask import json
from flask.wrappers import Request
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

class CodeSearchColBertController:

    def index(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        content = data.get("content", "")
        index_name = data.get("index_name", "")

        data = list(map(lambda x: x["content"], list(content.values())))
        collection = Collection(data=content)

        # collection = Collection(path="C:\\adverb-backend\\controllers\\downloads\\lotte\\science\\dev\\collection - Kopie.tsv")

        checkpoint = "C:\\adverb-backend\\models\\colbertv2.0"
        nbits = 2   # encode each dimension with 2 bits
        doc_maxlen = 300   # truncate passages at 300 tokens
        with Run().context(RunConfig(nranks=4)):
            config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)
            indexer = Indexer(checkpoint=checkpoint, config=config)
            indexer.index(name=index_name, collection=collection, overwrite=True)
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
        
        # return results