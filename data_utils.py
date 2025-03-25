from torch.utils.data import Dataset
import os
import json
import torch
from transformers import BartTokenizer
from sklearn.cluster import KMeans  # Added import for standard clustering
from sentence_transformers import SentenceTransformer
import random


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
        self.page_max_len = page_max_len
        self.is_test = is_test
        self.tgt_max_len = tgt_max_len
        self.num_pages = num_pages
        self.page_type = page_type
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')

    def __len__(self):
        return self.num

    def distribute_sentences(self, sentences):
        """
        Distribute sentences randomly across self.num_pages while maintaining order within each page.
        """
        num_sentences = len(sentences)
        base_count = num_sentences // self.num_pages
        extra = num_sentences % self.num_pages
        
        # Create a list of counts for each page ensuring equal distribution
        counts = [base_count + (1 if i < extra else 0) for i in range(self.num_pages)]
        
        # Randomly distribute indices into pages while preserving order
        indices = list(range(num_sentences))
        random.shuffle(indices)  # Shuffle indices
        
        pages = []
        start = 0
        for count in counts:
            page_indices = indices[start:start+count]  # Sort to maintain order
            pages.append([sentences[i] for i in page_indices])
            start += count
        
        return pages

    def __getitem__(self, idx):
        if self.isdir:
            with open(os.path.join(self.fdir, "%d.json"%idx), "rb") as f:
                data = json.load(f)
        else:
            with open(self.files[idx], "rb") as f:
                data = json.load(f)

        sentences = data["article"]
        pages = self.distribute_sentences(sentences)
        pages = [" ".join(page) for page in pages]
        
        src = self.tok.batch_encode_plus(pages, max_length=self.page_max_len, return_tensors="pt", padding="max_length", truncation=True)
        src_input_ids = src["input_ids"]
        abstract = data["abstract"]
        abstract = " ".join(abstract)
        tgt = self.tok.batch_encode_plus([abstract], max_length=self.tgt_max_len, return_tensors="pt", padding="max_length", truncation=True)
        tgt_input_ids = tgt["input_ids"]
        result = {
            "src_input_ids": src_input_ids, 
            "tgt_input_ids": tgt_input_ids,
            }
        if self.is_test:
            result["data"] = data
            result["data"]["pages"] = pages
        return result


def collate_mp(batch, pad_token_id, is_test=False):
    def mat_pad(X):
        seq_num = max([x.size(0) for x in X])
        result = torch.ones(len(X), seq_num, X[0].size(1), dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result

    src_input_ids = mat_pad([x["src_input_ids"] for x in batch])
    tgt_input_ids = torch.cat([x["tgt_input_ids"] for x in batch])
    if is_test:
        data = [x["data"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,
        "tgt_input_ids": tgt_input_ids,
        }
    if is_test:
        result["data"] = data
    return result



