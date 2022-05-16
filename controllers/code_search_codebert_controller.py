from controllers.printing import print_to_console
import os
from flask import json
from flask.wrappers import Request
import torch
from transformers import RobertaModel, RobertaTokenizer

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

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        model = RobertaModel.from_pretrained(os.path.join(os.getcwd(), "models", "codebert-base"), local_files_only=True)
        model.to(device)

        print_to_console("Search NL->PL - model:", "codebert")
        print_to_console("Search NL->PL - query:", search_text)

        query = search_text
        query_vec = model(tokenizer(query,return_tensors='pt').to(device).input_ids)[1]

        result = []
        for item in json.loads(str(content)):
            codePartsCounter = 0
            tensors = []
            lines = str(item["content"]).splitlines()
            i = 0
            while i < len(lines):
                codePartsCounter += 1
                code = lines[i : (i + batch_size)]
                code = " ".join(code).replace("\r\n", " ").replace("\n", " ")[:512]
                tokens = tokenizer(code, return_tensors="pt").to(device).input_ids
                print(str(i) + " " + item)
                code_vec = model(tokens)[1]
                tensors.append(code_vec)
                i += batch_size + 1

            code_vecs = torch.cat(tensors, 0)
            scores = torch.einsum("ab,cb->ac", query_vec, code_vecs)
            scores = torch.softmax(scores, -1)

            search_lines = []
            for i in range(codePartsCounter):
                score = scores[0, i].item()
                line = i * batch_size
                if score > 0.9:
                    search_lines.append(line)

            result.append({"index": item["index"], "match": search_text})

        print_to_console("Search NL->PL - result:", str(result))

        return { "result": {"search_text": search_text, "search_lines": search_lines, "batch_size": batch_size} }