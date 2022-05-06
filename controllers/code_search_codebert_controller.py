import os
from flask import json
from flask.wrappers import Request
import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
from tqdm import tqdm

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
        # model = RobertaModel.from_pretrained("python_model")

        query = search_text
        query_vec = model(tokenizer(query,return_tensors='pt').to(device).input_ids)[1]
        code_1=content[:512]
        code1_vec = model(tokenizer(code_1,return_tensors='pt').to(device).input_ids)[1]
        # code_2="s = 'hello world'"
        # code2_vec = model(tokenizer(code_2,return_tensors='pt').to(device).input_ids)[1]
        # code_3="hello world"
        # code3_vec = model(tokenizer(code_3,return_tensors='pt').to(device).input_ids)[1]
        #code_vecs=torch.cat((code1_vec),0) #,code2_vec,code3_vec
        codes = [code_1] #,code_2,code_3
        scores=torch.einsum("ab,cb->ac",query_vec,code1_vec)
        scores=torch.softmax(scores,-1)
        print("Query:",query)
        for i in range(3):
            print("Code:",codes[i])
            print("Score:",scores[0,i].item())

        return ""












        if not request.data:
            return None

        data = json.loads(request.data)
        content = data.get("content", "")
        search_text = data.get("search", "")
        batch_size = data.get("batch_size", 8)
        if not content or not search_text:
            return None

        if True:
            return self.new_search_implementation(content, search_text, batch_size)
        else:
            return self.old_search_implementation(content, search_text, batch_size)


    def new_search_implementation(self, content, search_text, batch_size):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        model = RobertaModel.from_pretrained(os.path.join(os.getcwd(), "models", "codebert-base"), local_files_only=True)
        # model = RobertaModel.from_pretrained("microsoft/codebert-base", config=config)
        model.to(device)
        query_vec = model(tokenizer(search_text, return_tensors="pt").to(device).input_ids)[1]
        codes = []
        tensors = []
        lines = content.splitlines()
        i = 0
        while i < len(lines):
            code = lines[i : (i + batch_size)]
            code = "\n".join(code)
            codes.append(code)
            code_vec =  model(tokenizer(code, return_tensors="pt").to(device).input_ids)[1]
            tensors.append(code_vec)
            i += (batch_size + 1)

        code_vecs = torch.cat(tensors, 0)
        scores = torch.einsum("ab,cb->ac", query_vec, code_vecs)
        scores = torch.softmax(scores, -1)
        
        # print("Query:", search_text)
        search_lines = []
        for i in range(len(codes)):
            score = scores[0, i].item()
            line = i * batch_size
            # print("Code:", codes[i])
            # print("Score:", score)
            if score > 0.75:
                search_lines.append(line)

        print(str(search_lines))

        return { "result": {"search_text": search_text, "search_lines": search_lines, "batch_size": batch_size} }
        
    def old_search_implementation(self, content, search_text, batch_size):
        max_seq_length = 200
        sequence_a_segment_id = 0
        sequence_b_segment_id = 1
        cls_token_segment_id = 1
        pad_token = 0
        pad_token_segment_id = 0

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        config = RobertaConfig.from_pretrained("microsoft/codebert-base", num_labels=2, finetuning_task="codesearch")
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", config=config)
        model.to(device)

        features = []
        lines = content.splitlines()
        i = 0
        while i < len(lines):
            search_text_tokens = tokenizer.tokenize(search_text)[:50]
            code = lines[i : (i + batch_size)]
            code = "\n".join(code)
            code_tokens = tokenizer.tokenize(code)
            self.truncate_seq_pair(search_text_tokens, code_tokens, max_seq_length-3)

            tokens = search_text_tokens + [tokenizer.sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            tokens += code_tokens + [tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(code_tokens) + 1)

            tokens = [tokenizer.cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
        
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=0))

            i += (batch_size + 1)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': None,
                          # XLM don't use segment_ids
                          'labels': batch[3]}

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        # preds_label = np.argmax(preds, axis=1)
        # print(np.min(list(map(lambda x: x[1], preds))), np.max(list(map(lambda x: x[1], preds))))
        preds_label = list(map(lambda x: 1 if x[1] > 0.2 else 0, preds))
        # all_logits = preds.tolist()

        search_lines = []
        for i in range(len(preds_label)):
            line = i * batch_size
            #text = "\n".join(lines[line : line + batch_size - 1])
            label = int(str(preds_label[i]))
            if label == 1:
                #search_lines.append({"line": line, "text": text, "label": label})
                search_lines.append(line)
        return { "result": {"search_text": search_text, "search_lines": search_lines, "batch_size": batch_size} }


    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id