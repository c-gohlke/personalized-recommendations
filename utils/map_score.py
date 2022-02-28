def map_score(pred, gt, k):
    total_score = 0
    for u in range(len(pred)):
        score = 0.0
        num_hits = 0.0
        if len(gt[u]) > 0:
            for i, p in enumerate(pred[u]):
                if p in gt[u] and p not in pred[u][:i]:
                    num_hits += 1.0
                    score += num_hits / (i + 1.0)
            total_score = total_score + (score / min(len(gt[u]), k))
    return total_score / len(pred)
