from controllers.printing import print_to_console
import os 
from flask import json
from flask.wrappers import Request
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Collection
from colbert import Indexer, Searcher
import torch
import shutil

class CodeSearchColBertController:

    def indexing(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        content = data.get("content", "")
        index_name = data.get("index_name", "adverb")
        if not content:
            return None
       
        print_to_console("Search indexing - model:", "colbert")
        content = json.loads(str(content))
        collection, _ = self.convert_json_to_collection(content)
        checkpoint = os.path.join(os.getcwd(), "models", "colbertv2.0")

        # FIX FOR IMPROVEMENT: manually deleting old index, otherwise have to wait for 20 seconds (overwrite bug...)
        index_folder = os.path.join(os.getcwd(), "experiments")
        if os.path.exists(index_folder) and os.path.isdir(index_folder):
            shutil.rmtree(index_folder, ignore_errors=True)

        nbits = 2   # encode each dimension with 2 bits
        doc_maxlen = 300   # truncate passages at 300 tokens
        nranks = 1 if torch.cuda.is_available() else 0  # number of gpu's to use
        with Run().context(RunConfig(nranks=nranks, overwrite=True, gpus=nranks)):
            config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)
            config.local_files_only = True  # use local indexer checkpoint
            config.overwrite = True
            indexer = Indexer(checkpoint=checkpoint, config=config)
            indexer.index(name=index_name, collection=collection, overwrite=True)
            # print_to_console(indexer.get_index())
        
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

        content = json.loads(str(content))
        collection, file_parts = self.convert_json_to_collection(content)
        checkpoint = os.path.join(os.getcwd(), "models", "colbertv2.0")
        
        nranks = 1 if torch.cuda.is_available() else 0  # number of gpu's to use
        with Run().context(RunConfig(nranks=nranks, gpus=nranks)):
            searcher = Searcher(index=index_name, checkpoint=checkpoint, collection=collection)

        results = searcher.search(query, k=10)
        
        model_result = {}
        for passage_id, passage_rank, passage_score in zip(*results):
            model_result[passage_id] = {"rank": passage_rank, "score": passage_score}
        print(str(model_result))
        part_index = 0
        for file in content:
            new_matches = []
            file_parts = file["matches"]
            for part in file_parts:
                if part_index in model_result:
                    new_matches.append(part)
                part_index += 1
            file["matches"] = new_matches
            print("Index: " + str(part_index))

        print_to_console("Search NL->PL - model:", "colbert")
        print_to_console("Search NL->PL - query:", query)
        print_to_console("Search NL->PL - result:", str(content))
        
        return content

    def convert_json_to_collection(self, content):
        data = []
        for file in content:
            file_parts = file["matches"]
            for part in file_parts:
                code = part["code"]
                data.append(code)

        collection = Collection(data=data)
        return collection, file_parts