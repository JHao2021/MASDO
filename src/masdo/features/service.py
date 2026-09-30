"""Observable 21-dimensional worker-task service inputs."""
import numpy as np

FEATURE_DIM = 21


def service_inputs(env, state):
    """Encode visible pairs without absolute worker/task coordinates.

    Concatenate five worker features, thirteen relative/service/metric pair
    features, and three service-pressure features. Inputs exclude the objective,
    worker identity, future tasks, and service labels.
    """
    count = len(state.task_ids)
    pair = state.pair_features[:, :count]
    workers = np.broadcast_to(state.worker_features[:, None, 2:],
                               (len(pair), count, 5))
    degree = state.feasible[:, :count].sum(axis=0).astype(np.float64)
    scarcity = np.divide(1., degree, out=np.zeros(count), where=degree > 0)
    coverage = 1. - pair[..., 10]
    urgency = np.array([1. / (max(0., env.task(t).deadline - env.time) + 1.)
                        for t in state.task_ids], dtype=np.float64)
    pressure = np.stack((np.broadcast_to(scarcity, coverage.shape), coverage,
                         np.broadcast_to(urgency, coverage.shape)), axis=-1)
    x = np.concatenate((workers, pair[..., 2:], pressure), axis=-1).astype(np.float32)
    if x.shape[-1] != FEATURE_DIM or not np.isfinite(x).all():
        raise ValueError('invalid observable service inputs')
    return x
