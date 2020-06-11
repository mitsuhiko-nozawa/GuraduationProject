from .trainer import trainer
from utils.util import cossim, rmse, mae

import numpy as np
import h5py
np.random.seed(0)

class mf_trainer(trainer):
    def __init__(self):
        self.param = None
        self.dim_size = None
        self.epoch_size = None
        self.train_mat = None
        self.valid_mat = None
        self.pred_mat = None
        self.test_mat = None
        self.user_size = None
        self.item_size = None
        self.U = None
        self.I = None
        self.mu = None
        self.bu = None
        self.bi = None
        self.step = None
        self.df = None
        self.valid_mae = []
        self.valid_rmse = []
        self.early_stopping_flag = None
        self.early_stopping = False


    def train(self, train_df, valid_df, dim_size, r, lr, epoch_size, step, early_stopping):
        # param config
        self.dim_size = dim_size
        self.epoch_size = epoch_size
        self.r = r
        self.lr = lr
        self.step = step
        self.early_stopping_flag = early_stopping

        # data preparation
        self.train_mat = self.to_matrix(train_df).values
        self.valid_mat = self.to_matrix(valid_df).values

        self.user_size = self.train_mat.shape[0]
        self.item_size = self.train_mat.shape[1]
        self.mu = np.sum(self.train_mat)/np.count_nonzero(self.train_mat)
        self.bu = np.sum(self.train_mat, axis=1) / np.count_nonzero(self.train_mat, axis=1)
        self.bi = np.sum(self.train_mat, axis=0) / np.count_nonzero(self.train_mat, axis=0)
        self.bu = self.bu.reshape(self.user_size, -1)
        self.bi = self.bi.reshape(-1, self.item_size)
        for i in range(0, self.bi.shape[1]):
            if self.bi[0][i] != self.bi[0][i]:
                self.bi[0][i] = 0.0

        for i in range(0, self.bu.shape[0]):
            if self.bu[i][0] != self.bu[i][0]:
                self.bu[i][0] = 0.0

        self.U = np.random.rand(self.user_size, self.dim_size)
        self.I = np.random.rand(self.item_size, self.dim_size)


        # train
        for epoch in range(0, self.epoch_size):
            self.pred_mat = (np.matmul(self.U, self.I.T) + self.mu + self.bu) + self.bi
            errors = self.train_mat - self.pred_mat # (user_size, itemsize)
            tempU = np.zeros([self.user_size, self.dim_size])
            tempI = np.zeros([self.item_size, self.dim_size])

            for i in range(0, self.user_size):
                for j in range(0, self.item_size):
                    tempU[i] += errors[i][j] * self.I[j]
                    tempI[j] += errors[i][j] * self.U[i]
            self.U = self.U + self.lr * (tempU - self.r * self.U)
            self.I = self.I + self.lr * (tempI - self.r * self.I)

            if (epoch+1) % self.step == 0:
                print("epoch ", epoch+1, " is done")
                train_rmse, train_mae = self.evaluate(flag="train")
                valid_rmse, valid_mae = self.evaluate(flag="valid")
                if train_rmse != train_rmse or train_mae != train_mae or valid_mae != valid_mae or valid_rmse != valid_rmse:
                    print("overflow is occuring")
                    print("stop training")
                    return 0
                #print("train: rmse is ", e1, ", mae is ", e2)
                #print("valid: rmse is ", e1_val, ", mae is ", e2_val)
                self.valid_mae.append(valid_mae)
                self.valid_rmse.append(valid_rmse)
                if self.early_stopping_flag and (epoch+1) != self.step \
                        and (self.valid_mae[-2] < self.valid_mae[-1] or self.valid_rmse[-2] < self.valid_rmse[-1]):
                    self.early_stopping = True

                if(epoch+1) == self.epoch_size or self.early_stopping:
                    mae_path = "models/losses/MF/loss_hp.hdf5"
                    try:
                        with h5py.File(mae_path, 'w') as f:
                            f.create_dataset("mae", data=self.valid_mae)
                            f.create_dataset("rmse", data=self.valid_rmse)
                            f.create_dataset('U', data=self.U)
                            f.create_dataset('I', data=self.I)
                    except:
                        print("can't find such directory")
                        with h5py.File("loss_hp.hdf5", 'w') as f:
                            f.create_dataset("mae", data=self.valid_mae)
                            f.create_dataset("rmse", data=self.valid_rmse)
                            f.create_dataset('U', data=self.U)
                            f.create_dataset('I', data=self.I)
                    return 0


    def predict(self, test_df):
        self.test_mat = self.to_matrix(test_df).values
        pass

    def evaluate(self, flag="test"):
        e1, e2 = None, None
        if flag == "train":
            e1, e2 = self.evaluate_(self.train_mat, self.pred_mat)
        elif flag == "valid":
            e1, e2 = self.evaluate_(self.pred_mat, self.valid_mat)
        else:
            e1, e2 = self.evaluate_(self.test_mat, self.pred_mat)


        print(flag, " : rmse is ", e1, ", mae is ", e2)
        return e1, e2









