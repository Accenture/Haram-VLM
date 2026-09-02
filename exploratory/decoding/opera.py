#!/usr/bin/env python
"""
OPERA over-trust penalty (Huang et al., CVPR 2024) for Qwen3-VL captioning.

OPERA observes that hallucination correlates with a "knowledge-aggregation" self-attention pattern:
over a window of recently generated tokens, the columns concentrate on a single summary position. OPERA
penalizes beam candidates that exhibit this pattern. We implement the core Over-Trust Penalty (OTP) in a
batched beam search using eager attention: each step we accumulate the last-layer, head-averaged
attention vector of the newly generated token; over a window of the last `k` such vectors we compute the
columnar aggregation score sigma = max_col prod_row(attn) (restricted to generated-text columns), and
subtract alpha*sigma from each beam's score before top-B selection. (We implement OTP; the optional
retrospection-allocation rollback is omitted and noted as such.)
"""
import torch


@torch.no_grad()
def opera_generate(model, proc, inp, device, max_new=64, num_beams=5, alpha=50.0, window=8,
                   eos_id=None, n_img_tokens=None):
    """Batched beam search with OPERA over-trust penalty. Returns the best beam's generated token ids."""
    tok = proc.tokenizer
    eos_id = eos_id if eos_id is not None else tok.eos_token_id
    B = num_beams

    # step 0: encode the prompt+image once, then expand to B beams
    out = model(**inp, use_cache=True, output_attentions=True)
    past = out.past_key_values
    logp = torch.log_softmax(out.logits[0, -1].float(), -1)               # [vocab]
    prompt_len = inp["input_ids"].shape[1]
    # length of generated-text region starts after the prompt; columns >= prompt_len are generated text
    topv, topi = logp.topk(B)
    beam_tokens = [[int(t)] for t in topi]                                 # per-beam generated ids
    beam_score = topv.clone()                                             # [B]
    # expand cache to B beams
    past.batch_repeat_interleave(B) if hasattr(past, "batch_repeat_interleave") else None
    cur = topi.view(B, 1)
    # per-beam rolling window of attention vectors (each [total_len])
    attn_win = [[] for _ in range(B)]
    # seed step-0 attention (the prompt's last position attention), head-averaged, for each beam
    a0 = out.attentions[-1][0].mean(0)[-1].float()                        # [prompt_len]
    for b in range(B): attn_win[b].append(a0.clone())

    finished = [False] * B
    for step in range(max_new - 1):
        o = model(input_ids=cur, past_key_values=past, use_cache=True, output_attentions=True)
        past = o.past_key_values
        logp = torch.log_softmax(o.logits[:, -1].float(), -1)             # [B, vocab]
        attn = o.attentions[-1].mean(1)[:, -1, :].float()                 # [B, total_len] head-avg, new-token row

        # over-trust penalty per beam from its attention window (restricted to generated-text columns)
        pen = torch.zeros(B, device=device)
        for b in range(B):
            attn_win[b].append(attn[b])
            if len(attn_win[b]) > window: attn_win[b].pop(0)
            if len(attn_win[b]) >= 3:
                L = min(v.shape[0] for v in attn_win[b])
                W = torch.stack([v[:L] for v in attn_win[b]], 0)          # [win, L]
                cols = W[:, prompt_len:L] if L > prompt_len else W        # generated-text columns
                if cols.shape[1] > 0:
                    sigma = cols.prod(0).max()                            # columnar aggregation score
                    pen[b] = sigma
        # candidate scores: beam_score + token_logp - alpha*penalty(beam)
        cand = beam_score.view(B, 1) + logp - alpha * pen.view(B, 1)      # [B, vocab]
        # top-B over all beams x vocab
        flat = cand.view(-1)
        topv, topf = flat.topk(B)
        parent = (topf // logp.shape[1]).tolist()
        token = (topf % logp.shape[1]).tolist()

        # rebuild beams
        new_tokens, new_attn = [], []
        for b in range(B):
            new_tokens.append(beam_tokens[parent[b]] + [int(token[b])])
            new_attn.append([v.clone() for v in attn_win[parent[b]]])
        beam_tokens = new_tokens; attn_win = new_attn
        beam_score = torch.tensor([beam_score[parent[b]].item() + logp[parent[b], token[b]].item()
                                   for b in range(B)], device=device)     # un-penalized cumulative logprob
        # reorder cache to parents
        beam_idx = torch.tensor([parent[b] for b in range(B)], device=device)
        past.reorder_cache(beam_idx) if hasattr(past, "reorder_cache") else past.reorder_cache(beam_idx)
        cur = torch.tensor([[token[b]] for b in range(B)], device=device)

        finished = [token[b] == eos_id or finished[parent[b]] for b in range(B)]
        if all(finished): break

    best = int(beam_score.argmax())
    ids = beam_tokens[best]
    if eos_id in ids: ids = ids[: ids.index(eos_id)]
    return ids
