import sklearn.preprocessing as pp


class DataProcesser:
    """ preprocess dataset """
    def __init__(self, n_start, n_train, n_test, target_idx):
        self.scaler = pp.MinMaxScaler()
        self.n_start = n_start
        self.n_train = n_train
        self.n_test = n_test
        self.idx = target_idx

    def load_data(self, raw_data):
        X = raw_data
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        X_train = X[self.n_start: self.n_start + self.n_train, :]
        Y_train = X_train[:, self.idx].reshape((-1, 1))
        Y_test = X[self.n_start + self.n_train: self.n_start + self.n_train + self.n_test, self.idx]
        return X_train, Y_train, Y_test
