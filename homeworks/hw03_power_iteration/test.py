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


def get_eigenvalues_and_eigenvectors_with_numpy(data):
    _eigenvalues, _eigenvectors = np.linalg.eig(data)
    max_index = np.argmax(np.abs(_eigenvalues))
    min_index = np.argmin(np.abs(_eigenvalues))

    _test_pair_a = np.array([_eigenvalues[max_index], _eigenvalues[min_index]])
    _test_pair_b = np.array([_eigenvectors[:, max_index], _eigenvectors[:, min_index]])
    if _test_pair_b[0][0] < 0:
        _test_pair_b[0] *= -1
    if _test_pair_b[1][0] < 0:
        _test_pair_b[1] *= -1

    return _test_pair_a, _test_pair_b


for _ in range(1000):
    size = np.random.choice(np.arange(2, 5))
    data = np.random.randn(size, size)
    data = data.T.dot(data)
    a0, b0 = get_dominant_eigenvalue_and_eigenvector(data, 1000)
    assert type(
        a0) == float, 'Return type for eigenvalue is not Python float (please, note, numpy.float64 is a different type)'
    assert type(b0) == np.ndarray, 'Return type for eigenvector is not np.ndarray'

    a1, b1 = get_dominant_eigenvalue_and_eigenvector(np.linalg.inv(data), 1000)
    a1 = 1 / a1

    if b0[0] < 0:
        b0 *= -1
    if b1[0] < 0:
        b1 *= -1

    assert np.allclose(data.dot(b0), a0 * b0,
                       atol=1e-3), f'Ax != \lambda x for the dominant eigenvalue check the solution!\n{data.dot(b0), a0 * b0}'
    assert np.allclose(data.dot(b1), a1 * b1,
                       atol=1e-3), f'Ax != \lambda x for the smallest eigenvalue check the solution!\n{data.dot(b1), a1 * b1}'

    _test_pair_a, _test_pair_b = get_eigenvalues_and_eigenvectors_with_numpy(data)

    assert np.allclose(_test_pair_a, np.array([a0, a1]),
                       atol=1e-3), f'Eigenvalues are different from np.linalg.eig!\n{_test_pair_a, np.array([a0, a1])}'
    assert np.allclose(_test_pair_b, np.array([b0, b1]),
                       atol=1e-3), f'Eigenvectors are different from np.linalg.eig!\n{_test_pair_b, np.array([b0, b1])}'

print(
    'Seems fine! Copy function `get_dominant_eigenvalue_and_eigenvector` to the .py file and submit your solution to the contest!')