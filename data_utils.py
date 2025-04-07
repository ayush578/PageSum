from torch.utils.data import Dataset
import os
import json
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer

def to_cuda(batch, gpuid):
    for n in batch:
        if n != "data":
            batch[n] = batch[n].to(gpuid)


class PageSumDataset(Dataset):
    def __init__(self, fdir, model_type, page_max_len=1024, tgt_max_len=256, num_pages=5, page_type=None, is_test=False):
        """ data format: article, abstract, label (optional) """
        self.isdir = os.path.isdir(fdir)
        if self.isdir:
            self.fdir = fdir
            self.num = len(os.listdir(fdir))
        else:
            with open(fdir) as f:
                self.files = [x.strip() for x in f]
            self.num = len(self.files)
        self.tok = BartTokenizer.from_pretrained(model_type, verbose=False)
        self.bart_model = BartForConditionalGeneration.from_pretrained(model_type)
        self.page_max_len = page_max_len
        self.is_test = is_test
        self.tgt_max_len = tgt_max_len
        self.num_pages = num_pages
        self.page_type = page_type

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if self.isdir:
            with open(os.path.join(self.fdir, "%d.json"%idx), "rb") as f:
                data = json.load(f)
        else:
            with open(self.files[idx], "rb") as f:
                data = json.load(f)

        num = len(data["article"]) // self.num_pages
        label = []
        for i in range(self.num_pages):
            label.extend([i] * num)
        while len(label) < len(data["article"]):
            label.append(self.num_pages - 1)
        
        article = [[] for _ in range(self.num_pages)]
        
        if self.page_type == "multi_doc":
            for i in range(min(len(data["article"]), self.num_pages)):
                article[i].append(data["article"][i])
        else:
            for (x, y) in zip(data["article"], label):
                article[y].append(x)

        for i in range(self.num_pages):
            if len(article[i]) == 0:
                article[i].append(".")

        try:
            article = [" ".join(x) for x in article]
        except:
            article = ["." for _ in range(self.num_pages)]
        
        # Encode the article pages as source inputs
        src = self.tok.batch_encode_plus(
            article,
            max_length=self.page_max_len,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        src_input_ids = src["input_ids"]

        # ----- Generate page-level summaries using BART -----
        # Here we assume self.bart_model and self.bart_tokenizer are already defined.
        page_summaries = []
        for page in article:
            # Tokenize the page text using the BART tokenizer
            inputs = self.tok(page, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            summary_ids = self.bart_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                early_stopping=True
            )
            page_summary = self.tok.decode(summary_ids[0], skip_special_tokens=True)
            page_summaries.append(page_summary)

        # ----- Divide the abstract into pages using the provided sentence list -----
        # Since data["abstract"] is already a list of sentences, we use it directly.
        abstract_sentences = data["abstract"]

        # Use the rouge_score package to compare sentences with each page summary.
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        page_assignment = {i: [] for i in range(self.num_pages)}
        for sent in abstract_sentences:
            best_page = None
            best_score = -1
            for i, ps in enumerate(page_summaries):
                score = scorer.score(sent, ps)['rougeL'].fmeasure
                if score > best_score:
                    best_score = score
                    best_page = i
            page_assignment[best_page].append(sent)

        # Build target summaries for each page; if no sentences assigned, use a fallback.
        page_targets = []
        for i in range(self.num_pages):
            if page_assignment[i]:
                page_targets.append(" ".join(page_assignment[i]))
            else:
                page_targets.append(".")

        # Encode the page-level summaries as target inputs.
        tgt = self.tok.batch_encode_plus(
            page_targets,
            max_length=self.tgt_max_len,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        tgt_input_ids = tgt["input_ids"]
        abstract = data["abstract"]
        abstract = " ".join(abstract)
        tgt = self.tok.batch_encode_plus([abstract], max_length=self.tgt_max_len, return_tensors="pt", padding="max_length", truncation=True)
        nett_tgt_input_ids = tgt["input_ids"]
        result = {
            "src_input_ids": src_input_ids, 
            "tgt_input_ids": tgt_input_ids,
            "nett_tgt_input_ids": nett_tgt_input_ids,
            }
        if self.is_test:
            result["data"] = data
            result["data"]["pages"] = article
        return result


def collate_mp(batch, pad_token_id, is_test=False):
    def mat_pad(X):
        seq_num = max([x.size(0) for x in X])
        result = torch.ones(len(X), seq_num, X[0].size(1), dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result

    src_input_ids = mat_pad([x["src_input_ids"] for x in batch])
    tgt_input_ids = mat_pad([x["tgt_input_ids"] for x in batch])
    nett_tgt_input_ids = torch.cat([x["nett_tgt_input_ids"] for x in batch])
    if is_test:
        data = [x["data"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,
        "tgt_input_ids": tgt_input_ids,
        "nett_tgt_input_ids": nett_tgt_input_ids,
        }
    if is_test:
        result["data"] = data
    return result



