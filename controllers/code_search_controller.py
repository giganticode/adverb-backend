from flask import json
from flask.wrappers import Request
import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
from tqdm import tqdm

class CodeSearchController:

    def search_for_text(self, request: Request):
        if not request.data:
            return None

        data = json.loads(request.data)
        content = data.get("content", "")
        search_text = data.get("search", "")
        batch_size = data.get("batch_size", 8)
        if not content or not search_text:
            return None
        
        max_seq_length = 200
        sequence_a_segment_id = 0
        sequence_b_segment_id = 1
        cls_token_segment_id = 1
        pad_token = 0
        pad_token_segment_id = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        print(np.min(list(map(lambda x: x[1], preds))), np.max(list(map(lambda x: x[1], preds))))
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