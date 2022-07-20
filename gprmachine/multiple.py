import numpy as np
import gpflow as gpf
from gpflow.ci_utils import ci_niter


class MultiGPR():
    """ implementation of Multi-task GPRMachine or MT-GPRM """
    def __init__(self, X_train, Y_train, n_iter, n_task):
        self.X_train = X_train                  # list of training X used in GPR
        self.Y_train = Y_train                  # list of training Y used in GPR
        self.n_iter = ci_niter(n_iter)
        self.n_task = n_task

    def set_init_kernel(self, lower, upper):
        np.random.seed()
        x_dim = np.shape(self.X_train[0])[1]
        lsv = np.random.uniform(lower, upper, x_dim)
        return x_dim, lsv

    def augment_xy(self):
        X_task = self.X_train[0]
        Y_task = self.Y_train[0]
        n_train = X_task.shape[0]
        mark = np.zeros((n_train, 1))
        X_augmented = np.hstack((X_task, mark))
        Y_augmented = np.hstack((Y_task, mark))
        for a in range(1, self.n_task):
            X_task = self.X_train[a]
            Y_task = self.Y_train[a]
            n_train = X_task.shape[0]
            mark = np.asarray([a] * n_train).reshape(n_train, 1)
            X_augmented = np.vstack((X_augmented, np.hstack((X_task, mark))))
            Y_augmented = np.vstack((Y_augmented, np.hstack((Y_task, mark))))
        return X_augmented, Y_augmented

    def build_model(self, k_low, k_up):
        output_dim = len(self.X_train)
        x_dim, lsv = self.set_init_kernel(k_low, k_up)
        dim_idx_list = [idx for idx in range(x_dim)]
        kernel_1 = gpf.kernels.Matern52(active_dims=dim_idx_list, lengthscales=lsv)
        kernel_2 = gpf.kernels.Matern32(active_dims=dim_idx_list, lengthscales=lsv)
        kernel_3 = gpf.kernels.RBF(active_dims=dim_idx_list, lengthscales=lsv)
        kernel_4 = gpf.kernels.White()
        base_kernel = kernel_1 + kernel_2 + kernel_3 + kernel_4
        coregion_kernel = gpf.kernels.Coregion(output_dim=output_dim,
                                               rank=output_dim,
                                               active_dims=[x_dim])
        kernel = base_kernel * coregion_kernel
        likelihood_list = [gpf.likelihoods.Gaussian() for _ in range(output_dim)]
        likelihood = gpf.likelihoods.SwitchedLikelihood(likelihood_list)
        X_augmented, Y_augmented = self.augment_xy()
        self.model = gpf.models.VGP((X_augmented, Y_augmented),
                                    kernel=kernel,
                                    likelihood=likelihood)
        self.coreg_kernel = coregion_kernel
        #print(self.model.trainable_parameters)

    def optimizing(self):
        gpf.optimizers.Scipy().minimize(self.model.training_loss,
                                        self.model.trainable_variables,
                                        options=dict(disp=False, maxiter=self.n_iter),
                                        method="L-BFGS-B")

    def predicting(self, X_test):
        self.mu = []
        self.var = []
        for t in range(self.n_task):
            x_test = X_test[t]
            n_test = x_test.shape[0]
            mark = np.asarray([t] * n_test).reshape((n_test, 1))
            X_augmented = np.hstack((x_test, mark))
            mu, var = self.model.predict_f(X_augmented)
            self.mu.append(mu.numpy().reshape(-1))
            self.var.append(var.numpy().reshape(-1))