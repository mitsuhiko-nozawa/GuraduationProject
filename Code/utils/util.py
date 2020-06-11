import numpy as np

def L2norm(a):
    return np.sum(a**2)**0.5

def cossim(a1, a2):
    norm = L2norm(a1) * L2norm(a2)
    if norm == 0.0:
        return 0.0
    else:
        return np.dot(a1,a2) / norm

def rmse(mat1, mat2): # mat1 is test_data, mat2 is pred_data
    # vailationとかでも使えるように、mat2で0のところは0にしたい
    error = 0
    for i in range(0, mat1.shape[0]):
        idxs = np.where(mat2[i] > 0)
        error += np.sum((mat1[i][idxs] - mat2[i][idxs])**2)
    return (error/np.count_nonzero(mat2))**0.5

def mae(mat1, mat2):
    error = 0
    for i in range(0, mat1.shape[0]):
        idxs = np.where(mat2[i] > 0)
        error += np.sum(np.abs((mat1[i][idxs] - mat2[i][idxs])))
    return (error/np.count_nonzero(mat2))

def to_matrix(self, df):
    if "reviewText" in df.columns:
        df.drop("reviewText", axis=1, inplace=True)
    df = df.pivot_table(values=["overall"], index=["reviewerID"], columns=["asin"])
    df.fillna(0, inplace=True)
    return df

#def EuclidanDistance():