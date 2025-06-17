import numpy as np
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from typing import Union
from sklearn import linear_model


class Estimator:

    def __init__(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray, pd.DataFrame]):
        """
        Initialize the Estimator with data.

        Parameters:
        X (Union[pd.DataFrame, np.ndarray]): Independent variables
        y (Union[pd.Series, np.ndarray, pd.DataFrame]): Dependent variable
        """
        if isinstance(X, pd.DataFrame):
            self.X = X.to_numpy()
        else:
            self.X = X

        if isinstance(y, (pd.Series, pd.DataFrame)):
            self.y = y.to_numpy()
        else:
            self.y = y

    def estimate_1NN(self):
        """
        Estimate using the 1NN method described by Devroye et al. (2018).

        Returns:
        float: The 1NN estimator value.
        """
        nn = NearestNeighbors(n_neighbors=2, metric='euclidean')
        nn.fit(self.X)
        distances, indices = nn.kneighbors(self.X)
        NN = indices[:, 1]
        m_1 = self.y[NN]
        n = len(self.y)
        S = np.dot(self.y, m_1) / n
        EY = np.dot(self.y, self.y) / n
        L = EY - S
        return L

    def estimate_LS(self):
        """
        Estimate variance using OLS.

        Returns:
        float: The estimated variance.
        """
        X_const = sm.add_constant(self.X)  # Adds a constant term to the predictors
        model = sm.OLS(self.y, X_const).fit()
        rss = sum(model.resid ** 2)
        degrees_of_freedom = len(self.y) - model.df_model - 1  # Minus 1 for the intercept
        variance = rss / degrees_of_freedom
        return variance
    
    def estimate_LASSO(self, K = 1):
        """ Computes an estimator for the noise level sigma^2 of the model via the scaled Lasso.

            **Parameters**

            *K*: ``float``. Constant in the definition. Defaults to 1, which is the choice from the scaled Lasso paper.
        """
        iter = 0
        max_iter = 50
        n_samples = self.X.shape[0]
        n_features = self.X.shape[1]
        tolerance = 1 / n_samples
        estimation_difference = 2 * tolerance
        alpha_0 = np.sqrt(K * np.log(n_features) / n_samples)

        lasso = linear_model.Lasso(alpha_0, fit_intercept = False)
        lasso.fit(self.X, self.y)
        noise_estimate = np.mean((self.y - lasso.predict(self.X))**2)

        while estimation_difference > tolerance and iter <= max_iter:
            alpha = np.sqrt(noise_estimate) * alpha_0
            lasso = linear_model.Lasso(alpha, fit_intercept = False)
            lasso.fit(self.X, self.y)

            new_noise_estimate = np.mean((self.y - lasso.predict(self.X))**2)
            estimation_difference = np.abs(new_noise_estimate - noise_estimate)
            noise_estimate = new_noise_estimate

            iter = iter + 1

        return noise_estimate

    def estimate(self, method='1NN', K = None):
        """
        General method to estimate based on the specified method.

        Parameters:
        method (str): The method to use for estimation ('1NN' or 'variance').

        Returns:
        float: The estimated value based on the specified method.
        """
        if method == '1NN':
            return self.estimate_1NN()
        elif method == 'LS':
            return self.estimate_LS()
        elif method == 'LASSO':
            return self.estimate_LASSO(K=K)
        else:
            raise ValueError("Unsupported method. Use '1NN' or 'LS'.")




