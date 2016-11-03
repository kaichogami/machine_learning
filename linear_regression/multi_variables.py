"""Linear Regression model for multiple features. Analytical and
   optimsation(gradiant) descent for weight estimation
"""
import numpy as np 


class MultiLinearRegression():
    """Linear Regression model with multiple features.
    y_bar = a + b1X1 + b2X2 + ... + bmXm

    J(b1, b2, b3....,bm) = (y - y_bar)^2

    We will find weights using gradiant descent and also
    analytically.

    Parameters
    ==========

    algorithm: string, "analytical", "gradiant"
      algorithm to use, to solve for weights
    intercept : bool
      True if intercept is to be calculated
    alpha: float, learning rate
      Learning rate in float
    max_iter: int
      max number of iterations for gradiant descent

    Attributes
    ==========
    weights_ : array of shape(n_samples)
       coefficients of hypothesis function.
    """

    def __init__(self, algorithm="gradiant", intercept=True, alpha=0.0001,
                 max_iter=10**3):
        self.algorithm = algorithm
        if algorithm == 'gradiant':
            if not(isinstance(alpha, float) or isinstance(max_iter, int)):
                raise ValueError("parameters not initialized properly")
        
        self.intercept = intercept
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        Fit the data to the model

        Parameters
        ==========
        X : array of shape(n_samples, features)
          The input array
        y : array of shape(n_samples)
          The labels of input

        Returns
        =======
        self : instance of self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Transpose X to design input matrix with rows as samples and
        # columns as features
        self.weights_ = np.zeros(X.shape[1])
        if self.intercept:
            self.intercept_ = 0

        if self.algorithm == 'gradiant':
            for _ in range(self.max_iter):
                self._gradiant_descent(X, y, X.shape[1])

        elif self.algorithm == 'analytic':
            self._analytic(X, y)

    def transform(self, X):
        """
        Transform the input matrix into predicted labels

        Parameters
        ==========
        X : array of shape(n_samples, n_features)
          The input matrix

        Returns
        =======
        y_bar : array of shape(n_samples)
          The predicted labes
        """
        X = np.asarray(X)
        y_bar = np.dot(X, self.weights_)
        if self.intercept:
            y_bar += self.intercept_
        return y_bar

    def _gradiant_descent(self, X, y, N):
        """Function to minimise the cost function using gradiant descent

        Parameter
        =========
        X : array of shape(n_samples, n_features)
          The input array
        y : array of shape(n_samples)
          The labels of the input
        N : int
          Size of the features
        """
        if self.intercept:
            error = y - (np.dot(X, self.weights_) + self.intercept_)
            temp_intercept = sum((2.0 / N) * (-1 * error))
            self.intercept_ = self.intercept_ - self.alpha * temp_intercept

        else:
            error = y - (self.weights_ * X)
        for i in xrange(N):
            delta_weight = sum((2.0 / N) * (-1 * error) * X[:, i])
            self.weights_[i] = self.weights_[i] - self.alpha * delta_weight

    def _analytic(self, X, y):
        """Function to calculate the weights of the model.
           W = (Xt * X)**-1 * Xt * y
           Xt : X transpose
           (Xt * X)**-1 : inverse of 

        Parameter
        =========
        X : array of shape(n_samples, n_features)
          The input array
        y : array of shape(n_samples)
          The labels of the input
        """
        A = np.dot(X.T, X)
        try:
            inverse_A = np.linalg.inv(A)
        except:
            raise ValueError("Looks like inverse is not possible, "
                             "try gradiant descent algorithm")
        
        self.weights_ = np.dot(np.dot(inverse_A, X.T), y)
