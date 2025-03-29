from torch.utils.data import Dataset
import os
import json
import torch
from transformers import BartTokenizer
import spacy
nlp = spacy.load("en_core_web_sm")

def to_cuda(batch, gpuid):
    for n in batch:
        if n != "data":
            batch[n] = batch[n].to(gpuid)

def preprocess_summary(document, summary):
    """
    Preprocess the summary so that tokens flagged as entities (using spaCy's NER)
    that are not present in the document are replaced with '<unk>'.
    Additionally, create a mask where the value is 1 for entity tokens.

    Parameters:
      document (str): The source document text.
      summary (str): The summary text.

    Returns:
      str: The processed summary with undesired entity tokens replaced by '<unk>'.
      list: A mask where 1 indicates an entity token and 0 otherwise.
    """
    # Process both document and summary using spaCy
    doc_doc = nlp(document)
    doc_summary = nlp(summary)
    
    # Create a set of allowed tokens from the document (using lowercase for case-insensitive matching)
    allowed_tokens = set(token.text.lower() for token in doc_doc)
    
    processed_tokens = []
    entity_mask = []  # Initialize the mask for entity tokens
    for token in doc_summary:
        # Check if token is part of a named entity (ent_iob_ is not "O" means it's either the beginning or inside an entity)
        if token.ent_iob_ != "O":
            # If the entity token (in lowercase) is not in the document, replace it with <unk>
            if token.text.lower() not in allowed_tokens:
                processed_tokens.append("<unk>")
            else:
                processed_tokens.append(token.text)
            entity_mask.append(1)  # Mark as an entity token
        else:
            processed_tokens.append(token.text)
            entity_mask.append(0)  # Not an entity token
    
    # Reconstruct the processed summary while preserving the original whitespace
    processed_summary = "".join([proc_tok + token.whitespace_ 
                                 for proc_tok, token in zip(processed_tokens, doc_summary)])
    return processed_summary, entity_mask

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
        
        src = self.tok.batch_encode_plus(article, max_length=self.page_max_len, return_tensors="pt", padding="max_length", truncation=True)
        src_input_ids = src["input_ids"]
        text = " ".join(data["article"])
        abstract = data["abstract"]
        abstract = " ".join(abstract)
        abstract, mask = preprocess_summary(text,abstract)
        tgt = self.tok.batch_encode_plus([abstract], max_length=self.tgt_max_len, return_tensors="pt", padding="max_length", truncation=True)
        tgt_input_ids = tgt["input_ids"]
        result = {
            "src_input_ids": src_input_ids, 
            "tgt_input_ids": tgt_input_ids,
            "mask": mask,
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

    def pad_masks(masks, max_len, pad_value=0):
        padded_masks = [mask[:max_len] + [pad_value] * (max_len - len(mask[:max_len])) for mask in masks]
        return torch.tensor(padded_masks, dtype=torch.long)

    src_input_ids = mat_pad([x["src_input_ids"] for x in batch])
    tgt_input_ids = torch.cat([x["tgt_input_ids"] for x in batch])
    max_len = tgt_input_ids.size(1)  # Use the max length of tgt_input_ids
    masks = pad_masks([x["mask"] for x in batch], max_len)

    if is_test:
        data = [x["data"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,
        "tgt_input_ids": tgt_input_ids,
        "mask": masks,
    }
    if is_test:
        result["data"] = data
    return result



