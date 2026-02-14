import multiprocessing as mp
import os
import time

import numpy as np


def generate_data(n_samples=1_000_000, n_features=20, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features), dtype=np.float64)
    true_w = rng.standard_normal(n_features)
    logits = x @ true_w
    y = (logits > 0).astype(np.float64)
    return x, y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sequential_sgd_no_shuffle(x, y, epochs=8, lr=0.1, batch_size=8192):
    n_samples, n_features = x.shape
    w = np.zeros(n_features, dtype=np.float64)
    for _ in range(epochs):
        for start in range(0, n_samples, batch_size):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            preds = sigmoid(xb @ w)
            grad = (xb.T @ (preds - yb)) / len(xb)
            w -= lr * grad
    return w


X_G = None
Y_G = None


def init_worker(x, y):
    global X_G, Y_G
    X_G = x
    Y_G = y


def grad_for_range(args):
    s, e, w = args
    xb = X_G[s:e]
    yb = Y_G[s:e]
    preds = sigmoid(xb @ w)
    return (xb.T @ (preds - yb)) / len(xb)


def distributed_sgd_sync(x, y, epochs=8, base_lr=0.1, batch_size=8192, workers=4, warmup_epochs=3):
    n_samples, n_features = x.shape
    w = np.zeros(n_features, dtype=np.float64)
    scaled_lr = base_lr * workers

    # Use spawn for stability with NumPy/BLAS on macOS.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=init_worker, initargs=(x, y)) as pool:
        for epoch in range(epochs):
            lr = scaled_lr * ((epoch + 1) / warmup_epochs) if epoch < warmup_epochs else scaled_lr

            for start in range(0, n_samples, batch_size * workers):
                tasks = []
                for wid in range(workers):
                    s = start + wid * batch_size
                    e = min(s + batch_size, n_samples)
                    if s >= n_samples:
                        break
                    tasks.append((s, e, w))

                grads = pool.map(grad_for_range, tasks)
                w -= lr * np.mean(grads, axis=0)

    return w


def main():
    # Make BLAS single-thread to expose process-level speedup.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    n_samples = 1_000_000
    n_features = 20
    epochs = 8
    batch_size = 8192
    workers = min(4, mp.cpu_count())

    x, y = generate_data(n_samples=n_samples, n_features=n_features)

    t0 = time.time()
    w_seq = sequential_sgd_no_shuffle(x, y, epochs=epochs, lr=0.1, batch_size=batch_size)
    t_seq = time.time() - t0

    t0 = time.time()
    w_par = distributed_sgd_sync(
        x, y, epochs=epochs, base_lr=0.1, batch_size=batch_size, workers=workers
    )
    t_par = time.time() - t0

    acc_seq = np.mean((sigmoid(x @ w_seq) > 0.5) == y)
    acc_par = np.mean((sigmoid(x @ w_par) > 0.5) == y)

    print(f"Samples: {n_samples}, Features: {n_features}, Epochs: {epochs}, Batch: {batch_size}, Workers: {workers}")
    print(f"Sequential time: {t_seq:.4f}s | acc: {acc_seq:.4f}")
    print(f"Parallel time:   {t_par:.4f}s | acc: {acc_par:.4f}")
    print(f"Speedup: {t_seq / t_par:.4f}x")


if __name__ == "__main__":
    main()
