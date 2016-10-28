import numpy as np

from numpy.testing import (assert_array_equal)

from ..single_variable import LinearRegression

old_settings = np.seterr(all='ignore')

def test_linear_regression():

    X = np.random.randint(1, 10000, 1000)
    y = np.random.randint(1, 10000, 1000)
    y = y * 100 + 2 * 20 + 1000

    lr = LinearRegression(0.0001, 0.3)
    lr.fit(X[:800], y[:800])
    y_bar = lr.transform(X[800:])
    assert_array_equal(y.shape, y_bar.shape)

