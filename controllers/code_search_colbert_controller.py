from controllers.printing import print_to_console
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
        batch_size = data.get("batch_size", 8)
        if not content:
            return None
       
        print_to_console("Search indexing - model:", "colbert")
        collection, _ = self.convert_json_to_collection(content, batch_size)
        checkpoint = os.path.join(os.getcwd(), "models", "colbertv2.0")

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
        batch_size = data.get("batch_size", 8)
        if not query or not content:
            return None

        collection, file_parts = self.convert_json_to_collection(content, batch_size)
        checkpoint = os.path.join(os.getcwd(), "models", "colbertv2.0")
        
        nranks = 1 if torch.cuda.is_available() else 0  # number of gpu's to use
        with Run().context(RunConfig(nranks=nranks, gpus=nranks)):
            searcher = Searcher(index=index_name, checkpoint=checkpoint, collection=collection)

        results = searcher.search(query, k=10)
        
        return_values = []
        for passage_id, passage_rank, passage_score in zip(*results):
            file_part = file_parts[passage_id]
            match = {"line": file_part["line"], "rank": passage_rank, "score": passage_score}
            found = False
            for v in return_values:
                if v["relativePath"] == file_part["relativePath"]:
                    v["match"].append(match)
                    found = True
                    break
            if found == False:
                return_values.append({ "relativePath": file_part["relativePath"], "match": [match] })

        print_to_console("Search NL->PL - model:", "colbert")
        print_to_console("Search NL->PL - query:", query)
        print_to_console("Search NL->PL - result:", str(return_values))
        
        return return_values

    def convert_json_to_collection(self, content, batch_size):
        content = json.loads(str(content))
        data = []
        file_parts = []
        for item in content:
            file_content = str(item["content"])
            if file_content:
                file_content = file_content.replace("\r\n", " ").replace("\n", " ")
                lines = file_content.splitlines()
                i = 0
                while i < len(lines):
                    code = lines[i : (i + batch_size)]
                    file_parts.append({"relativePath": item["relativePath"], "line": i})
                    data.append(str(code))
                    i += batch_size + 1
        print_to_console("Search NL->PL - len data:", str(len(data)))
        print_to_console("Search NL->PL - len file_parts:", str(len(file_parts)))
        collection = Collection(data=data)
        return collection, file_parts