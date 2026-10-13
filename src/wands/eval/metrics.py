import math
RELEVANCE = {"Exact": 2, "Partial": 1, "Irrelevant": 0}

def ndcg_at_k(predicted_ids, relevance, k):
    dcg = dcg_at_k(predicted_ids, relevance, k)
    best = idcg(relevance, k)
    return dcg/best if best > 0 else 0

def idcg(relevance, k):
    k = min(k, len(relevance))
    res = 0
    relevance = sorted(relevance.values(), reverse=True)
    for i in range(k):
        res+=(relevance[i] / math.log2(i+2))

    return res

def dcg_at_k(predicted_ids, relevance, k):
    k = min(k, len(predicted_ids))
    res = 0
    for i in range(k):
        p_id = predicted_ids[i]
        res+=(relevance.get(p_id, 0) / math.log2(i + 2))

    return res

def mrr(predicted_ids, relevance):
    for i in range(len(predicted_ids)):
        if relevance.get(predicted_ids[i], 0) > 0:
            return 1/(i+1)
    return 0

def recall_at_k(predicted_ids, relevance, k):
    mink = min(k, len(predicted_ids))
    topk = 0
    for i in range(mink):
        if relevance.get(predicted_ids[i], 0) > 0:
            topk+=1

    total = sum(1 for v in relevance.values() if v > 0)

    return topk/total if total > 0 else 0