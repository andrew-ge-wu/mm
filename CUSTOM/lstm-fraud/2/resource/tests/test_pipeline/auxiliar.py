import numpy as np


def generate_logistic(coefficients, samples, random_state=None):
    """
    Generates `samples` samples from the logistic latent process from N=len(coefficients) features.

    The result is a np.array of the form `[[x1,...,xN,y],...]`
    """
    if random_state is not None:
        np.random.seed(random_state)
    coefficients = np.array(coefficients)
    data = []
    for sample in range(samples):
        values = np.random.normal(size=len(coefficients))
        noise = np.random.logistic()

        c = (np.sum(values*coefficients) + noise > 0).astype(np.int64)
        data.append(np.append(values, c))
    return np.array(data)
