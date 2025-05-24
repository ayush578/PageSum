import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
# from compare_mt.rouge.rouge_scorer import RougeScorer
from rouge_score import rouge_scorer
from transformers import BartTokenizer
from utils import Recorder
from data_utils import to_cuda, collate_mp, PageSumDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
import logging
import nltk
from nltk import sent_tokenize
from modeling_bart_ours import PageSumModel
from transformers import Adafactor
from config import arxiv, arxiv_discourse, pubmed, govreport, multinews, base_setting
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Subset


logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)


class label_smoothing_loss(nn.Module):
    def __init__(self, ignore_index, epsilon=0.1):
        super(label_smoothing_loss, self).__init__()
        self.ignore_idx = ignore_index
        self.epsilon = epsilon

    def forward(self, input, target):
        input = input.transpose(1, 2) # [batch_size, seq_len, word_num]
        input = torch.log_softmax(input, dim=2)
        k = input.size(2)
        target_prob = torch.ones_like(input).type_as(input) * self.epsilon * 1 / k
        mask = torch.arange(k).unsqueeze(0).unsqueeze(0).expand(target.size(0), target.size(1), -1).type_as(target)
        mask = torch.eq(mask, target.unsqueeze(-1).expand(-1, -1, k))
        target_prob.masked_fill_(mask, 1 - self.epsilon + (self.epsilon * 1 / k))
        loss = - torch.mul(target_prob, input)
        loss = loss.sum(2)
        # mask ignore_idx
        mask = (target != self.ignore_idx).type_as(input)
        loss = torch.mul(loss, mask).sum() / mask.sum()
        return loss


def evaluation(args):
    # load data
    if args.config == "arxiv":
        arxiv(args)
    elif args.config == "arxiv_discourse":
        arxiv_discourse(args)
    elif args.config == "pubmed":
        pubmed(args)
    elif args.config == "govreport":
        govreport(args)
    elif args.config == "multinews":
        multinews(args)
    else:
        base_setting(args)
    tok = BartTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)
    test_set = PageSumDataset(f"/home/ubuntu/mtp/PageSum/{args.dataset}/{args.datatype}/test", args.model_type, is_test=True, page_max_len=args.page_max_len, tgt_max_len=args.tgt_max_len, num_pages=args.num_pages, page_type=args.page_type)
    dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    scorer = PageSumModel.from_pretrained(model_path, gradient_checkpointing=args.gradient_checkpointing, use_cache=not args.gradient_checkpointing)
    if args.cuda:
        scorer = scorer.to("cuda:0")

    if args.model_dir:
        scorer.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pth")))
    
    for name, param in scorer.named_parameters():
        if "encoder" in name:
            param.data = param.data.to("cuda:1")
        if "decoder" in name:
            param.data = param.data.to("cuda:0")
    scorer.eval()

    model_name = args.model_dir.split("/")[-1]

    print(model_name)
    
    cnt = 0
    rouge_score = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = 0, 0, 0
    scorer.set_seq_num(args.num_pages)
    do_generate = True

    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            if (i%args.test_limit==0 and i!=0):
                break
            if args.cuda:
                to_cuda(batch, args.gpuid[0])
            input_ids = batch["src_input_ids"]
            input_ids = input_ids.view(input_ids.size(0), -1)
            input_mask = input_ids != tok.pad_token_id
            eos_token_id = torch.tensor([scorer.get_config().eos_token_id], device="cuda:0")
            if do_generate:
                summaries = scorer.generate(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    eos_token_id = eos_token_id,
                    max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                    min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                    no_repeat_ngram_size=3,
                    length_penalty=2,
                    early_stopping=False,
                    seq_num=args.num_pages,
                )
                dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                for (sample, d) in zip(batch["data"], dec):
                    sents = sent_tokenize(d)
                    score = rouge_score.score("\n".join(sample["abstract"]), "\n".join(sents))
                    rouge1 += score["rouge1"].fmeasure
                    rouge2 += score["rouge2"].fmeasure
                    rougeL += score["rougeL"].fmeasure
                    cnt += 1    
            print(f"batch: {i}")
            print(f"rouge1: {rouge1/cnt}, rouge2: {rouge2/cnt}, rougeL: {rougeL/cnt}")


def test(dataloader, scorer, args, gpuid, tok):
    scorer.eval()
    cnt = 0
    rouge_score = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = 0, 0, 0
    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            if (i%args.test_limit==0 and i!=0):
                break
            if args.cuda:
                to_cuda(batch, args.gpuid[0])
            input_ids = batch["src_input_ids"]
            input_ids = input_ids.view(input_ids.size(0), -1)
            input_mask = input_ids != tok.pad_token_id
            eos_token_id = torch.tensor([scorer.get_config().eos_token_id], device="cuda:0")
            summaries = scorer.generate(
                input_ids=input_ids,
                attention_mask=input_mask,
                eos_token_id = eos_token_id,
                max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                early_stopping=True,
                seq_num=args.num_pages
            )
            dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
            for (sample, d) in zip(batch["data"], dec):
                sents = sent_tokenize(d)
                score = rouge_score.score("\n".join(sample["abstract"]), "\n".join(sents))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeL += score["rougeL"].fmeasure
                cnt += 1
        rouge1 = rouge1 / cnt
        rouge2 = rouge2 / cnt
        rougeL = rougeL / cnt
        scorer.train()
        return rouge1, rouge2, rougeL
    
def print_gpu_usage():
    print(f"GPU {0}: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB allocated, {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB reserved")
    print(f"GPU {1}: {torch.cuda.memory_allocated(1) / 1024**2:.2f} MB allocated, {torch.cuda.memory_reserved(1) / 1024**2:.2f} MB reserved")

def run(args):
    if args.config == "arxiv":
        arxiv(args)
    elif args.config == "arxiv_discourse":
        arxiv_discourse(args)
    elif args.config == "pubmed":
        pubmed(args)
    elif args.config == "govreport":
        govreport(args)
    elif args.config == "multinews":
        multinews(args)
    else:
        base_setting(args)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(f"cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    tok = BartTokenizer.from_pretrained(args.model_type)

    start_idx = args.start * args.batch_size
    end_idx = (args.end + 1) * args.batch_size
    # Create a subset
    subset_indices = list(range(start_idx, end_idx))

    # collate_fn_val = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)
    # val_set = PageSumDataset(f"/scratch/aayush.cse20.itbhu/mtp/PageSum/{args.dataset}/val", args.model_type, is_test=True, page_max_len=args.page_max_len, tgt_max_len=args.tgt_max_len, num_pages=args.num_pages, page_type=args.page_type)
    # val_dataloader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn_val)
    collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)
    train_set = PageSumDataset(f"/scratch/pushpa.rs.cse23.itbhu/mtp_ayush/PageSum/arxiv/base/train", args.model_type, is_test=True, page_max_len=args.page_max_len, tgt_max_len=args.tgt_max_len, num_pages=args.num_pages, page_type=args.page_type)
    train_subset = Subset(train_set, subset_indices)
    dataloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    scorer = PageSumModel.from_pretrained(model_path, gradient_checkpointing=args.gradient_checkpointing, use_cache=not args.gradient_checkpointing)
    if args.model_dir:
        scorer.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pth")))
    if args.cuda:
        scorer = scorer.to("cuda:0")
    
    for name, param in scorer.named_parameters():
        if "encoder" in name:
            param.data = param.data.to("cuda:1")
        if "decoder" in name:
            param.data = param.data.to("cuda:0")
        # print(f"Layer: {name} | Device: {param.device}")
    
    scorer.train()
    mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    init_lr = args.max_lr / args.warmup_steps
    if args.optim == "adafactor":
        s_optimizer = Adafactor(scorer.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
    else:
        s_optimizer = optim.Adam(scorer.parameters(), lr=init_lr)
    all_step_cnt = 0
    scorer.set_seq_num(args.num_pages)

    # start training
    print("epochs",args.epoch)
    print(min(len(dataloader), args.end-args.start))
    for epoch in range(args.epoch):
        print("epoch: ",epoch)
        print("Current Time:", datetime.now().strftime("%H:%M:%S"),flush=True)
        s_optimizer.zero_grad()
        step_cnt = 0
        for (i, batch) in enumerate(dataloader):
            # if i<=args.start:
            #     continue
            # if i>args.end:
            #     break
            if args.cuda:
                to_cuda(batch, 0)
            step_cnt += 1
            raw_src = batch["src_input_ids"]
            input_ids = raw_src.view(raw_src.size(0), -1)
            input_mask = input_ids != tok.pad_token_id
            decoder_input_ids = batch["tgt_input_ids"]
            decoder_attention_mask = decoder_input_ids != tok.pad_token_id
            gold = decoder_input_ids[:, 1:]
            orig_seq_num = 7               # remember full-document setting
            scorer.set_seq_num(1)
            page_summaries = []
            # print_gpu_usage()
            scorer.eval()
            with torch.no_grad():
                eos_token_id = torch.tensor([scorer.get_config().eos_token_id], device="cuda:0")
                for p in range(args.num_pages):
                    pid = raw_src[:, p, :].clone()
                    pmask = (pid != tok.pad_token_id).long()
                    gen_ids = scorer.generate(
                        input_ids=pid,
                        attention_mask=pmask,
                        eos_token_id = eos_token_id,
                        max_length=args.gen_max_len,
                        min_length=args.gen_min_len,
                        no_repeat_ngram_size=3,
                        length_penalty=args.length_penalty,
                        early_stopping=True,
                        seq_num=1
                    )
                    page_summaries.append(tok.decode(gen_ids[0], skip_special_tokens=True))
                scorer.set_seq_num(orig_seq_num)
            
                reference = " ".join(batch["data"][0]["abstract"])
                texts = [reference] + page_summaries
                embs = embedder.encode(texts, convert_to_tensor=True)
                ref_emb = embs[0:1]                                        # (1, dim)
                page_embs = embs[1:] 
                sims = cosine_similarity(page_embs.cpu().numpy(),
                                        ref_emb.cpu().numpy()).reshape(-1)
                sims = np.clip(sims, a_min=0, a_max=None)
                teacher_w = sims / sims.sum()
                teacher_w = torch.tensor(teacher_w, device=input_ids.device)
                # print_gpu_usage()
                student_s = scorer.attention_score(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask
                )
                student_w = student_s.mean(dim=1).mean(dim=0)               # → (num_pages,)
                # print_gpu_usage()
            scorer.train() 
            # 4) compute auxiliary KL loss
            weight_loss = F.kl_div(student_w.log(), teacher_w, reduction="batchmean") 
            output = scorer(
                input_ids=input_ids, 
                attention_mask=input_mask,
                decoder_input_ids=decoder_input_ids, 
                decoder_attention_mask=decoder_attention_mask,
                output_hidden_states=False
                )
            output = output[0]
            output = output[:, :-1]  # truncate last token
            # gold = batch["tgt_input_ids"][:, 1:]  # shift right
            mle_loss = mle_fn(output.transpose(1, 2), gold)
            loss = mle_loss + args.lambda_w * weight_loss 
            loss = loss / args.accumulate_step
            loss.backward()
                
            if step_cnt == args.accumulate_step:
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(scorer.parameters(), args.grad_norm)
                step_cnt = 0
                all_step_cnt += 1
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in s_optimizer.param_groups:
                    param_group['lr'] = lr
                s_optimizer.step()
                s_optimizer.zero_grad()
            del loss, output

            if (i%args.cycle==0):
                print("input: ",i+args.start)
                print("Current Time:", datetime.now().strftime("%H:%M:%S"),flush=True)
                if args.save_dir :
                    torch.save(scorer.state_dict(), f"./{args.save_dir}/model.pth")
                # if args.do_generate:
                #     rouge1, rouge2, rougeL = test(val_dataloader, scorer, args, gpuid, tok)
                #     loss = 1 - 2 * rouge1 * rouge2 / (rouge1 + rouge2)
                #     print(f"rouge1: {rouge1}, rouge2: {rouge2}, rougeL: {rougeL}")
                # else:
                #     loss = test(val_dataloader, scorer, args, gpuid, tok)
                # print("Current Time:", datetime.now().strftime("%H:%M:%S"),flush=True)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("-e", "--evaluate", action="store_true", help="evaluate")
    parser.add_argument("-l", "--log", action="store_true", help="log")
    parser.add_argument("-p", "--port", type=int, default=12355, help="port")
    parser.add_argument("--model_pt", default="", type=str, help="model path")
    parser.add_argument("--config", default="base", type=str, help="config path")
    parser.add_argument("--start", type=int, default=0, help="strting index in dataset")
    parser.add_argument("--end", type=int, default=100000, help="ending index in dataset")
    parser.add_argument("--test_limit", type=int, default=500, help="test limit")
    parser.add_argument("--cycle", type=int, default=100, help="no of samples after which you want to show and save results")
    parser.add_argument("--save_dir", type=str, default=None, help="Path for the trained model to save it")
    parser.add_argument("--model_dir", type=str, default=None, help="Path for the trained model to load it")
    parser.add_argument("--lambda_w",type=float,default=1.0,help="strength of the auxiliary page‐weight KL loss")
    args = parser.parse_args()

    if args.cuda is False:
        if args.evaluate:
            evaluation(args)
        else:
            run(args)
    else:
        if args.evaluate:
            with torch.cuda.device("cuda:0"):
                evaluation(args)
        else:
            with torch.cuda.device("cuda:0"):
                run(args)