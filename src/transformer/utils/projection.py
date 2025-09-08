import numpy as np
def linear_projection(X, Y):
    return (np.dot(X,Y) / np.linalg.norm(Y, keepdims=True, axis=-1))