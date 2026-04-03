
"""
surrogate/active_learning.py

Active learning loop for surrogate-guided NAS.

Algorithm:
  1. Sample a large random pool of chromosomes
  2. Surrogate predicts val_accuracy for all of them (cheap)
  3. Take top-K predicted architectures → fully evaluate them (expensive)
  4. Add results to training data → retrain surrogate
  5. Repeat for N rounds

This cuts expensive evaluations by (1 - top_k_ratio) per round.
"""
import random, time
import numpy as np
from search_space.chromosome import decode_chromosome, chromosome_to_features
from training.proxy_trainer import proxy_train

GENE_BOUNDS = [
    (1,4),(0,5),(0,5),(0,5),(0,5),(0,2),(0,2),(1,3),(0,4),(0,3),(0,1),(0,1)
]

def _random_chromosome():
    return [random.randint(lo, hi) for lo, hi in GENE_BOUNDS]


def run_active_learning_round(
    surrogate,
    X_train     : np.ndarray,
    y_train     : np.ndarray,
    train_loader,
    val_loader,
    device,
    pool_size   : int   = 200,
    top_k       : int   = 5,
    proxy_epochs: int   = 3,
    lr          : float = 1e-3,
    round_num   : int   = 1,
    verbose     : bool  = True,
) -> dict:
    t0 = time.time()
    if verbose:
        print(f"\n  ── Active Learning Round {round_num} ──")
        print(f"     Pool size: {pool_size}  |  Top-K evaluated: {top_k}  |  Saved: {pool_size - top_k} evals")

    pool_chroms = [_random_chromosome() for _ in range(pool_size)]
    X_pool      = np.array([chromosome_to_features(c) for c in pool_chroms], dtype=np.float32)

    pred_mean, pred_std = surrogate.predict_with_std(X_pool)
    acquisition_scores = pred_mean
    top_k_idx          = np.argsort(acquisition_scores)[::-1][:top_k]
    top_k_chroms       = [pool_chroms[i] for i in top_k_idx]

    if verbose:
        print(f"     Surrogate predicted top-{top_k} range: {acquisition_scores[top_k_idx[-1]]:.4f} – {acquisition_scores[top_k_idx[0]]:.4f}")

    X_new, y_new     = [], []
    evaluated_results = []

    for rank, chrom in enumerate(top_k_chroms):
        model  = decode_chromosome(chrom)
        result = proxy_train(model, train_loader, val_loader, device, epochs=proxy_epochs, lr=lr)
        feats  = chromosome_to_features(chrom)
        X_new.append(feats)
        y_new.append(result['val_accuracy'])
        evaluated_results.append({
            'round'       : round_num,
            'rank'        : rank + 1,
            'chromosome'  : chrom,
            'pred_acc'    : float(acquisition_scores[top_k_idx[rank]]),
            'true_acc'    : result['val_accuracy'],
            'num_params'  : result['num_params'],
            'train_time'  : result['train_time'],
        })
        if verbose:
            pred = acquisition_scores[top_k_idx[rank]]
            true = result['val_accuracy']
            print(f"     [{rank+1}/{top_k}] pred={pred:.4f}  true={true:.4f}  params={result['num_params']:,}")

    elapsed = time.time() - t0
    return {
        'X_new': np.array(X_new, dtype=np.float32),
        'y_new': np.array(y_new, dtype=np.float32),
        'evaluated_results': evaluated_results,
        'n_pool': pool_size,
        'n_evaluated': top_k,
        'n_saved': pool_size - top_k,
        'elapsed_s': elapsed,
    }

def run_active_learning(surrogate, X_seed, y_seed, train_loader, val_loader, device, n_rounds=3, pool_size=200, top_k=5, proxy_epochs=3, lr=1e-3, min_r2=0.3, verbose=True):
    X_train, y_train = X_seed.copy(), y_seed.copy()
    all_round_results, surrogate_metrics = [], []
    total_saved, total_evaluated = 0, len(y_seed)

    for rnd in range(1, n_rounds + 1):
        metrics = surrogate.fit(X_train, y_train)
        surrogate_metrics.append({'round': rnd, **metrics})
        round_result = run_active_learning_round(surrogate, X_train, y_train, train_loader, val_loader, device, pool_size, top_k, proxy_epochs, lr, rnd, verbose)
        all_round_results.append(round_result)
        X_train = np.vstack([X_train, round_result['X_new']])
        y_train = np.concatenate([y_train, round_result['y_new']])
        total_saved += round_result['n_saved']
        total_evaluated += round_result['n_evaluated']

    all_evaluated = [res for rnd_res in all_round_results for res in rnd_res['evaluated_results']]
    best_result = max(all_evaluated, key=lambda x: x['true_acc'])
    return {
        'X_train': X_train, 'y_train': y_train, 'round_results': all_round_results,
        'surrogate_metrics': surrogate_metrics, 'best_result': best_result,
        'total_evaluated': total_evaluated, 'total_saved': total_saved,
        'saving_pct': 100 * total_saved / (total_saved + total_evaluated)
    }
