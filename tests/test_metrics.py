import math

from wands.eval.metrics import dcg_at_k, idcg, mrr, ndcg_at_k, recall_at_k


# Shared test data
PREDICTED = [101, 102, 103, 104, 105]
RELEVANCE = {101: 2, 102: 0, 103: 1, 104: 0, 105: 2}


def test_dcg_at_k():
    # position 0: rel=2, position 1: rel=0, position 2: rel=1
    expected = 2 / math.log2(2) + 0 / math.log2(3) + 1 / math.log2(4)
    assert abs(dcg_at_k(PREDICTED, RELEVANCE, k=3) - expected) < 1e-9


def test_idcg():
    # Best ordering of relevance values [2, 2, 1, 0, 0] → top 3: [2, 2, 1]
    expected = 2 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert abs(idcg(RELEVANCE, k=3) - expected) < 1e-9


def test_ndcg_at_k():
    dcg = 2 / math.log2(2) + 0 / math.log2(3) + 1 / math.log2(4)
    ideal = 2 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    expected = dcg / ideal
    assert abs(ndcg_at_k(PREDICTED, RELEVANCE, k=3) - expected) < 1e-9


def test_ndcg_perfect_ranking():
    # If we rank all relevant items first: [101, 105, 103, 102, 104]
    perfect = [101, 105, 103, 102, 104]
    assert abs(ndcg_at_k(perfect, RELEVANCE, k=5) - 1.0) < 1e-9


def test_ndcg_no_relevant():
    relevance = {101: 0, 102: 0}
    assert ndcg_at_k([101, 102], relevance, k=2) == 0


def test_mrr_first_is_relevant():
    assert mrr(PREDICTED, RELEVANCE) == 1.0


def test_mrr_third_is_first_relevant():
    # 201 and 202 are not in relevance (treated as 0), 103 has rel=1
    assert mrr([201, 202, 103], RELEVANCE) == 1 / 3


def test_mrr_no_relevant():
    assert mrr([201, 202], RELEVANCE) == 0


def test_recall_at_k():
    # In top 3: 101 (rel=2) and 103 (rel=1) are relevant → 2 found
    # Total relevant: 101 (2), 103 (1), 105 (2) → 3 total
    assert abs(recall_at_k(PREDICTED, RELEVANCE, k=3) - 2 / 3) < 1e-9


def test_recall_at_k_all_found():
    assert abs(recall_at_k(PREDICTED, RELEVANCE, k=5) - 1.0) < 1e-9


def test_recall_no_relevant():
    relevance = {101: 0, 102: 0}
    assert recall_at_k([101, 102], relevance, k=2) == 0
