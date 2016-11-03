import numpy as np

from numpy.testing import (assert_array_equal)
from sklearn.preprocessing import StandardScaler

from ..multi_variables import MultiLinearRegression
from ...utils.scaling import standardize

def test_multilinearRegression():

    X = np.random.randint(1, 1000, (100, 10))
    y = np.asarray([sum(X[i]) * int(np.random.random_integers(1, 1000, 1)) + 1000
                    for i in xrange(100)])
    
    sc = StandardScaler().fit(X)
    X = sc.transform(X)
    mlr = MultiLinearRegression()
    mlr.fit(X, y)
    y_bar = mlr.transform(X)

