import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    eigenvector = np.random.rand(data.shape[1])
    if eigenvector[0] == 0:
        eigenvector[0] = 0.1

    for k in range(num_steps):
        eigenvector = np.dot(data, eigenvector) / np.linalg.norm(np.dot(data, eigenvector))

    eigenvalue = (np.dot(data, eigenvector) / eigenvector)
    eigenvalue = eigenvalue.tolist()[0]


    return eigenvalue, eigenvector