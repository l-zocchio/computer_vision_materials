import numpy as np

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

def one_hot_encode(labels, nclasses):
    l_shape = list(labels.shape)
    newshape = np.concatenate((l_shape, [nclasses]))
    onehot = np.zeros(newshape).astype(np.int8)
    for coordinates in cartesian([np.array(range(shape)) for shape in l_shape]):
        one_val = np.concatenate((coordinates, [int(labels[tuple(coordinates)])])) 
        onehot[tuple(one_val)] = 1
    return onehot

def one_hot_decode(label):
    l_shape = list(label.shape[:-1])
    decoded = np.zeros(l_shape).astype(np.int8)
    for coordinates in cartesian([np.array(range(dim)) for dim in l_shape]):
        decoded[tuple(coordinates)] = np.argmax(label[tuple(coordinates)])
    return np.squeeze(decoded)