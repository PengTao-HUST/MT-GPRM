import numpy as np
from .single import GeneralGPR
from .multiple import MultiGPR

class SingleTrainer:
    """ training process of GPRM """
    def __init__(self, X_train, Y_train, n_test, dropout, n_run, target):
        self.X_train = X_train # traning input data
        self.Y_train = Y_train # traning target data
        self.n_train = np.shape(X_train)[0]
        self.n_test = n_test # predicted length
        self.n_map = n_test + 1
        self.dropout = dropout
        self.n_run = n_run
        self.X_dim = X_train.shape[1]
        self.target = target # the index of target variable in input data

    def get_kernels(self, kernel_flag):
        if kernel_flag == 'BasicTrain':
            return self.kernels_BT
        elif kernel_flag == 'ConsisTrain':
            return self.kernels_CT
        else:
            print('Wrong flag was input, please manually check.')

    def variable_selection(self, idx_keep=None):
        if idx_keep is None:
            indx_keep = []
            for kernel in self.kernels_PT:
                lsv = kernel.get_params()['k1__k2__length_scale']
                indx_keep_temp = [i for i in range(self.X_dim) if lsv[i] < 100]
                indx_keep = np.append(indx_keep, indx_keep_temp)
            indx_keep = list(set(list(indx_keep)))
        else:
            indx_keep = idx_keep

        indx_all = [i for i in range(self.X_dim)]
        indx_del = [item for item in indx_all if item not in indx_keep]
        X_train = np.delete(self.X_train, indx_del, axis=1)
        return X_train

    def pre_training(self):
        print('>> Pre-training is in processing ...')
        kernels = []
        for m in range(0, self.n_map):
            X_train = self.X_train[:self.n_train-m, ]
            Y_train = self.Y_train[m:,]
            GPR = GeneralGPR(X_train, Y_train, self.dropout, self.n_run)
            GPR.fit()
            kernel = GPR.get_kernel()
            kernels.append(kernel)
        self.kernels_PT = kernels

    def basic_training(self):
        print('>> Basic training is in processing ...')
        kernels = []
        for m in range(0, self.n_map):
            print('>> Now training for %d-th mapping ...'%(m))
            Xs_train = self.variable_selection()
            X_train = Xs_train[:self.n_train-m, ]
            Y_train = self.Y_train[m:,]
            GPR = GeneralGPR(X_train, Y_train, self.dropout, self.n_run)
            GPR.fit()
            kernel = GPR.get_kernel()
            kernels.append(kernel)
        self.kernels_BT = kernels

    def consistent_training(self):
        print('>> Consistent training is in processing ...')
        kernels = [i for i in range(self.n_map)]
        pred_y_mean_list = []
        for ps in range(0, self.n_test):
            print('>> Now training with %d-th sample ...'%(ps))
            pred_Y = []
            for m in range(ps+1, self.n_map):
                Xs_train = self.variable_selection()
                X_train = Xs_train[:self.n_train-m+ps, ]
                Y_train = np.append(self.Y_train[m:,], pred_y_mean_list[:ps])
                X = Xs_train[self.n_train-m+ps,]
                GPR = GeneralGPR(X_train, Y_train, self.dropout, self.n_run)
                GPR.fit()
                kernel = GPR.get_kernel()
                kernels[m] = kernel
                pred_y = GPR.Predict(X)
                pred_Y.append(pred_y)
            pred_y_mean = np.average(pred_Y)
            pred_y_mean_list.append(pred_y_mean)
        self.kernels_CT = kernels


class MultipleTrainer:
    """ training process of MT-GPRM """
    def train(self, X_train, Y_train, n_test, n_task, n_iter, k_low_list, k_up_list):
        mu_list = [] # mean value 
        var_list = [] # variation
        n_train = X_train.shape[0]
        print('===' * 25)
        for g in range(0, n_test):
            print('    >>> Training process is running for group %d in total of %d ...' % (g + 1, n_test))
            X_Train = []
            Y_Train = []
            X_Test = []
            for t in range(n_task):
                X_task = X_train[:n_train - g - t - 1, :]
                X_Train.append(X_task)
                Y_task = Y_train[g + t + 1:, :]
                Y_Train.append(Y_task)
                X_test = X_train[n_train - g - t - 1:, :]
                X_Test.append(X_test)
            k_low = k_low_list[g]
            k_up = k_up_list[g]
            
            # MT-GPRM
            gpr_model = MultiGPR(X_Train, Y_Train, n_iter, n_task)
            gpr_model.build_model(k_low, k_up)
            gpr_model.optimizing()
            gpr_model.predicting(X_Test)
            
            for t in range(n_task):
                if len(gpr_model.mu[t]) <= n_test:
                    mu_list.append(gpr_model.mu[t])
                    var_list.append(gpr_model.var[t])
                else:
                    mu_list.append(gpr_model.mu[t][:n_test])
                    var_list.append(gpr_model.var[t][:n_test])
                    
        self.mu_list = mu_list
        self.var_list = var_list
        mean_MU, mean_VAR = self.calculate_mean(n_test, mu_list, var_list)
        print('===' * 25)
        return mean_MU, mean_VAR

    @staticmethod
    def calculate_mean(n_test, mu_list, var_list):
        """ average predictions from different task """
        MU = []
        VAR = []
        for i in range(0, len(mu_list)):
            current_mu = mu_list[i]
            current_var = var_list[i]
            n_mu_var = len(current_mu)
            if n_mu_var < n_test:
                expent_temp = np.asarray(['None' for _ in range(n_test - n_mu_var)])
                MU.append(np.hstack((current_mu, expent_temp)))
                VAR.append(np.hstack((current_var, expent_temp)))
            else:
                MU.append(current_mu[:n_test])
                VAR.append(current_var[:n_test])
        MU = np.asarray(MU)
        VAR = np.asarray(VAR)
        mean_MU = []
        mean_VAR = []
        for t in range(n_test):
            p_mu = list(MU[:, t])
            while 'None' in p_mu:
                p_mu.remove('None')
            p_var = list(VAR[:, t])
            while 'None' in p_var:
                p_var.remove('None')
            mean_MU.append(np.average(np.asarray(p_mu, dtype=float)))
            mean_VAR.append(np.average(np.asarray(p_var, dtype=float)))
        return np.asarray(mean_MU), np.asarray(mean_VAR)

