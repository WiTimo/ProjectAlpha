import logging
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch


def _collect_logistic_samples(loader, max_samples: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    collected = 0

    for batch in loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        last_step = x[:, :, -1].detach().cpu().numpy()
        batch_features = last_step.reshape(last_step.shape[0], -1)
        batch_labels = y[:, 0].detach().cpu().numpy().astype(np.int64)

        features.append(batch_features)
        labels.append(batch_labels)
        collected += len(batch_labels)

        if collected >= max_samples:
            break

    if not features:
        return None

    X = np.concatenate(features, axis=0)[:max_samples]
    y = np.concatenate(labels, axis=0)[:max_samples]
    return X, y


def train_logistic_baseline(loader, config: dict) -> Optional[LogisticRegression]:
    eval_cfg = config.get("evaluation", {})
    max_samples = int(eval_cfg.get("logistic_max_samples", 200_000))
    max_iter = int(eval_cfg.get("logistic_max_iter", 200))
    C = float(eval_cfg.get("logistic_C", 1.0))
    solver = eval_cfg.get("logistic_solver", "saga")
    random_state = int(eval_cfg.get("logistic_random_state", 1337))

    sample_result = _collect_logistic_samples(loader, max_samples)
    if sample_result is None:
        logging.warning("Logistic baseline skipped: no samples collected")
        return None

    X, y = sample_result
    if X.size == 0 or len(np.unique(y)) < 2:
        logging.warning("Logistic baseline skipped: insufficient class diversity")
        return None

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X, y)

    up_rate = float(np.mean(y)) * 100.0
    logging.info(
        "Logistic baseline trained on %d samples | Positive rate: %.2f%% | Solver=%s",
        len(y),
        up_rate,
        solver,
    )
    return model
