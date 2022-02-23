from flask import json
from flask.wrappers import Request
from transformers import RobertaTokenizer, T5ForConditionalGeneration, RobertaForMaskedLM, pipeline

class CodeSymbolController:

    def get_symbol_name(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        text = data.get("content", "")
        if not text:
            return None
        
        model_type = data.get("modelType", 2)

        if model_type == 0:
            model = RobertaForMaskedLM.from_pretrained("huggingface/CodeBERTa-small-v1")
            tokenizer = RobertaTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
            fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
            result = fill_mask(text)
            result = result[0]["token_str"].strip()
        elif model_type == 1:
            model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
            tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
            fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
            result = fill_mask(text)
            result = result[0]["token_str"].strip()
        else:
            tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
            model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
            input_ids = tokenizer(text, return_tensors="pt").input_ids
            generated_ids = model.generate(input_ids, max_length=8)
            result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return {"result": result}