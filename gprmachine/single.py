import random
import numpy as np
import scipy.optimize
from scipy.linalg import cholesky, cho_solve
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class GeneralGPR():
    """ implementation of GPRMachine or GPRM """
    def __init__(self, X_train, Y_train, dropout, n_run):
        self.X_train = X_train
        self.Y_train = Y_train
        self.dropout = dropout
        self.n_run = n_run

    def set_init_kernel(self):
        np.random.seed()
        x_dim = np.shape(self.X_train)[1]
        sigma = np.random.uniform(1500, 2000)
        lsv = np.random.uniform(1500, 2000, x_dim)
        self.kernel = sigma ** 2 * RBF(length_scale=lsv) + WhiteKernel()

    def set_init_kernel_(self):
        np.random.seed()
        x_dim = np.shape(self.X_train)[1]
        sigma = np.random.uniform(1500, 2000)
        lsv = np.random.uniform(1500, 2000, x_dim)
        self.kernel_ = sigma ** 2 * RBF(length_scale=lsv) + WhiteKernel()

    def get_kernel(self):
        return self.kernel

    def fit(self):
        best_lml = np.inf
        n_run = 0
        self.set_init_kernel()
        while True:
            n_run += 1
            self.set_init_kernel_()
            optima_theta, lml_value = self.optimization(self.kernel_.theta,
                                                        self.kernel_.bounds,
                                                        self.log_marginal_likelihood)

            if lml_value < best_lml:
                best_lml = lml_value
                self.kernel.theta = optima_theta
            if n_run == self.n_run:
                break

    def log_marginal_likelihood(self, theta):
        # Set updated kernel
        kernel = self.kernel_
        kernel.theta = theta

        # Compute K_inv*y
        K, K_gradient = kernel(self.X_train, eval_gradient=True)
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta))
        if self.Y_train.ndim == 1:
            Y_train = self.Y_train[:, np.newaxis]
        alpha = cho_solve((L, True), Y_train)

        # Compute log-likelihood
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", Y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)

        # Compute gradient
        np.random.seed()
        tmp = np.einsum("ik,jk->ijk", alpha, alpha)
        tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
        lml_gradient_dims = 0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)
        lml_gradient = lml_gradient_dims.sum(-1)
        lml_gradient_dropout = lml_gradient.copy()
        gradient_dim = len(lml_gradient)
        dropout_mask = sorted(random.sample(range(1, gradient_dim),
                                            int(self.dropout * gradient_dim)))
        for i in dropout_mask:
            lml_gradient_dropout[i] = 0.0
        return -log_likelihood, -lml_gradient_dropout

    def optimization(self, initial_theta, bounds, obj_func):
        opt_res = scipy.optimize.minimize(obj_func,
                                          initial_theta,
                                          method="L-BFGS-B",
                                          jac=True,
                                          bounds=bounds)
        theta_opt, lml_opt = opt_res.x, opt_res.fun
        return theta_opt, lml_opt

    def Predict(self, X):
        K = self.kernel(self.X_train)
        L = cholesky(K, lower=True)
        alpha = cho_solve((L, True), self.Y_train)
        K_trans = self.kernel(X, self.X_train)
        y_mean = K_trans.dot(alpha)
        return y_mean[0]
