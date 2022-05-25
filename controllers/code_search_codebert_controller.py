from controllers.printing import print_to_console
import os
from flask import json
from flask.wrappers import Request
import torch
from transformers import RobertaModel, RobertaTokenizer
from controllers.unixcoder import UniXcoder

class CodeSearchCodeBertController:
    
    def search_for_text(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        content = data.get("content", "")
        search_text = data.get("search", "")
        batch_size = data.get("batch_size", 8)
        if not content or not search_text:
            return None

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = UniXcoder("microsoft/unixcoder-base")
        model.to(device)

        print_to_console("Search NL->PL - model:", "codebert")
        print_to_console("Search NL->PL - query:", search_text)

        query = search_text

        tokens_ids = model.tokenize([query],max_length=512,mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(device)
        _, query_embedding = model(source_ids)
        norm_query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)

        result = []
        for item in json.loads(str(content)):
            relativePath = str(item["relativePath"])
            print_to_console("Search NL->PL - document:", relativePath)
            codePartsCounter = 0
            lines = str(item["content"]).splitlines()
            i = 0
            file_results = []
            while i < len(lines):
                codePartsCounter += 1
                code = lines[i : (i + batch_size)]
                code = " ".join(code).replace("\r\n", " ").replace("\n", " ")[:512]
                tokens_ids = model.tokenize([code], max_length=512, mode="<encoder-only>")
                source_ids = torch.tensor(tokens_ids).to(device)
                _, code_embedding = model(source_ids)
                norm_code_embedding = torch.nn.functional.normalize(code_embedding, p=2, dim=1)
                similarity = torch.einsum("ac,bc->ab", norm_code_embedding, norm_query_embedding)
                score = similarity[0, 0].item()
                print_to_console("Search NL->PL - line:", str(i) + " - " + str(score))
                if similarity > 0.9:
                    file_results.append({"line": i, "score": score })
                i += batch_size + 1

            result.append({"relativePath": relativePath, "match": file_results})
        print_to_console("Search NL->PL - result:", "Done!")
        return result


    def search_for_text2(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        content = data.get("content", "")
        search_text = data.get("search", "")
        batch_size = data.get("batch_size", 8)
        if not content or not search_text:
            return None

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        model = RobertaModel.from_pretrained(os.path.join(os.getcwd(), "models", "codebert-base"), local_files_only=True)
        model.to(device)

        print_to_console("Search NL->PL - model:", "codebert")
        print_to_console("Search NL->PL - query:", search_text)

        query = search_text
        query_vec = model(tokenizer(query,return_tensors="pt").to(device).input_ids)[1]

        result = []
        for item in json.loads(str(content)):
            print_to_console("Search NL->PL - document:", str(item["relativePath"]))
            codePartsCounter = 0
            tensors = tuple()
            lines = str(item["content"]).splitlines()
            i = 0
            while i < len(lines):
                codePartsCounter += 1
                code = lines[i : (i + batch_size)]
                code = " ".join(code).replace("\r\n", " ").replace("\n", " ")[:512]
                tokens = tokenizer(code, return_tensors="pt").to(device).input_ids
                code_vec = model(tokens)[1]
                tensors = tensors + (code_vec,)
                i += batch_size + 1

            code_vecs = torch.cat(tensors, 0)
            scores = torch.einsum("ab,cb->ac", query_vec, code_vecs)
            scores = torch.softmax(scores, -1)

            search_lines = []
            for i in range(codePartsCounter):
                score = scores[0, i].item()
                line = i * batch_size
                if score > 0.9:
                    search_lines.append({"line": line, "score": score})
            if len(search_lines) > 0:
                print_to_console("Search NL->PL - document matches:", search_lines)
                result.append({"relativePath": item["relativePath"], "match": search_lines})

        print_to_console("Search NL->PL - result:", "Done!")

        return result