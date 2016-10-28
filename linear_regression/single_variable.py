"""Simple implementation of Linear regression model
   with gradiant descent optimisation.
"""
import numpy as np

class LinearRegression():
    """Finds a line which best fits the training data
    of a single i/p variable.
    y = a + bX
    we find the best value of a and b applying gradiant
    descent on a cost function given by
    J(a, b) = (Y - a + bX)^2

    Gradiant Descent:
    a = a - alpha * delta(J(a, b))
    b = b - alpha * delta(J(a, b))
    while keeping b and a constant respectively.

    Parameters
    ==========

    alpha: float, learning rate
       Learning rate in float.
    stopping_condition: int, defaults to 5.
       int indicates the time to stop. 1<t<10
    """

    def __init__(self, alpha, stopping_condition = 5):
        self.alpha = alpha
        self.stopping_condition = stopping_condition

    def fit(self, X, y):
        """
        Fit the data on the linear regression model, i.e fit a line
        in the given data.

        Parameters
        ==========
        X : array of shape (n_samples)
          Independent input data.
        y : array of shape (n_samples)
          Output data, y = f(X)
        """
        X = np.asarray(X)
        if X.ndim != 1:
            raise ValueError("Only single feature input is supported")

        # initialise theta as zero.
        self.theta = [0, 0]
        if self.stopping_condition > 10:
            raise ValueError("Time should be less than 10 secs")

        self._gradiant_descent(X, y)


    def transform(self, X):
        """
        Transform X to f(x) given the linear regression parameters.

        Parameters
        ==========
        X : array of shape(n_samples)
          The input
        """
        X = np.asarray(X)
        if X.ndim != 1:
            raise ValueError("Only single feature input is supported")

        X *= self.theta[1]
        X += self.theta[0]
        return X

    def _gradiant_descent(self, X, y):
        """
        Gradiant descent optmization method to minimise cost function with
        parameters theta0 and theta1.

        Parameters
        ==========
        X : array of shape(n_samples)
          The input
        y : array of shape(n_samples)
          The output, y = f(x)
        """
        import time

        start = time.time()
        N = len(X)
        while True:
            delta_theta0 = sum(((2.0 / N) * (-y + (self.theta[0] + (self.theta[1] * X)))))
            delta_theta1 = sum(((2.0 / N) * X * (-y + (self.theta[0] + (self.theta[1] * X)))))
            self.theta[0] = self.theta[0] - self.alpha  * delta_theta0
            self.theta[1] = self.theta[1] - self.alpha  * delta_theta1
            if (time.time() - start >= self.stopping_condition):
                break

    def plot(self, X, y, y_pred):
        """
        Plot graph of X, real y and predicted y as a line

        Parameters
        ==========
        X : array, 1-D
         The input feature vector
        y : array, 1-D
         The output
        """

        from matplotlib import pyplot as plt

        plt.plot(X, y, 'rs', X, y_pred)
        plt.show()
