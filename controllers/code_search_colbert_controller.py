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

        data = {}
        for key in content:
            file_content = str(content[key]["content"])
            if file_content:
                file_content = file_content.replace("\r\n", " ").replace("\n", " ")
                id = int(content[key]["id"])
                data[id] = file_content
        collection = Collection(data=data)
        # collection = Collection(path=os.path.join(os.getcwd(), "downloads", "lotte", "lifestyle", "dev", "collection.tsv"))

        # path = os.path.join(os.getcwd(), "downloads", "lotte", "lifestyle", "dev", "collection_adverb.tsv")
        # if os.path.exists(path):
        #     os.remove(path)

        # with open(path, "a", encoding="utf-8") as file:
        #     for key in content:
        #         file_content = str(content[key]["content"])
        #         if file_content:
        #             file_content = file_content.replace("\r\n", " ").replace("\n", " ")
        #             line = str(content[key]["id"]) + "\t" + file_content + "\n"
        #             file.write(line)

        # collection = Collection(path=path)

        checkpoint = os.path.join(os.getcwd(), "models", "colbertv2.0")

        nbits = 2   # encode each dimension with 2 bits
        doc_maxlen = 300   # truncate passages at 300 tokens
        nranks = 1 if torch.cuda.is_available() else 0 # number of gpu's to use
        with Run().context(RunConfig(nranks=nranks)):
            config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)
            config.local_files_only = True
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
        
        return ""