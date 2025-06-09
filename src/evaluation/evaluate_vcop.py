import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import kendalltau, spearmanr
import itertools

def evaluate_vcop_all(model, X_test, y_test, config):
    """
    VCOP 평가: Top-1 Accuracy, Top-k Accuracy, Kendall's Tau, Spearman's Rho
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = TensorDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # 순열 리스트 (예: 4! = 24개)
    perms = list(itertools.permutations(range(config.tuple_len)))

    total_correct = 0
    total_topk_correct = 0
    total_exact = 0
    total_kendall = 0
    total_spearman = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)

            preds = outputs.argmax(dim=1)
            topk_preds = torch.topk(outputs, k=config.topk, dim=1).indices
            match_matrix = topk_preds.eq(targets.view(-1, 1))

            total_correct += (preds == targets).sum().item()
            total_topk_correct += match_matrix.any(dim=1).sum().item()

            # Permutation 유사도 평가
            for true_idx, pred_idx in zip(targets, preds):
                true_perm = perms[true_idx.item()]
                pred_perm = perms[pred_idx.item()]

                tau, _ = kendalltau(true_perm, pred_perm)
                rho, _ = spearmanr(true_perm, pred_perm)

                total_kendall += tau
                total_spearman += rho
                total_exact += int(true_perm == pred_perm)
                total_samples += 1

    top1_acc = total_correct / total_samples
    topk_acc = total_topk_correct / total_samples
    avg_kendall = total_kendall / total_samples
    avg_spearman = total_spearman / total_samples
    exact_match_acc = total_exact / total_samples

    print("\n🧪 [VCOP 평가 결과]")
    print(f"✅ Top-1 정확도        : {top1_acc:.4f}")
    print(f"✅ Top-{config.topk} 정확도     : {topk_acc:.4f}")
    print(f"📐 평균 Kendall's Tau : {avg_kendall:.4f}")
    print(f"📐 평균 Spearman Rho  : {avg_spearman:.4f}")
    print(f"🎯 완전 일치 비율     : {exact_match_acc:.4f}")

    return {
        "top1_acc": top1_acc,
        "topk_acc": topk_acc,
        "kendall": avg_kendall,
        "spearman": avg_spearman,
        "exact_match": exact_match_acc
    }