from controllers.printing import print_to_console
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
        query = data.get("search", "")
        if not content or not query:
            return None

        content = json.loads(str(content))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = UniXcoder("microsoft/unixcoder-base")
        model.to(device)

        tokens_ids = model.tokenize([query],max_length=512,mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(device)
        _, query_embedding = model(source_ids)
        norm_query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)

        result = []
        for file in content:
            new_matches = []
            file_parts = file["matches"]
            for part in file_parts:
                code = part["code"].strip()[:512].strip()
                tokens_ids = model.tokenize([code], max_length=512, mode="<encoder-only>")
                source_ids = torch.tensor(tokens_ids).to(device)
                _, code_embedding = model(source_ids)
                norm_code_embedding = torch.nn.functional.normalize(code_embedding, p=2, dim=1)
                similarity = torch.einsum("ac,bc->ab", norm_code_embedding, norm_query_embedding)
                score = similarity[0, 0].item()
                if similarity > 0.4:
                    new_matches.append(part)
            file["matches"] = new_matches

        print_to_console("Search NL->PL - model:", "codebert")
        print_to_console("Search NL->PL - query:", query)
        print_to_console("Search NL->PL - result:", str(content))
        
        return content
