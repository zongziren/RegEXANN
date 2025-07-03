import sys

def load_results(path):
    with open(path, 'r') as f:
        return [list(map(int, line.strip().split())) for line in f if line.strip()]

def evaluate_recall(gt_file, output_file, topk):
    gt_list = load_results(gt_file)
    pred_list = load_results(output_file)

    assert len(gt_list) == len(pred_list), "Mismatch in number of queries"
    recalls = []

    for i, (gt, pred) in enumerate(zip(gt_list, pred_list)):
        gt_set = set(gt[:topk])
        pred_set = set(pred[:topk])

        correct = len(gt_set & pred_set)
        padding = max(0, topk - len(pred))
        recall = (correct + padding) / topk

        status = f" ({correct}+{padding}/{topk})"
        recalls.append(recall)
        print(f"Query {i}: Recall@{topk} = {recall:.2f}{status}")

    avg_recall = sum(recalls) / len(recalls)
    print(f"\nAverage Recall@{topk}: {avg_recall:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python evaluate_recall.py <groundtruth.txt> <output.txt> <K>")
        sys.exit(1)

    gt_file = sys.argv[1]
    out_file = sys.argv[2]
    K = int(sys.argv[3])
    evaluate_recall(gt_file, out_file, K)
