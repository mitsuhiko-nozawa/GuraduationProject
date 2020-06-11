import numpy as np
import pandas as pd
from .trainer import trainer
from utils.util import cossim
np.set_printoptions(precision=3)


class cf_trainer(trainer):
    def __init__(self):
        self.df = None # rate matrix
        self.train_mat = None # np.array
        self.valid_mat = None
        self.test_mat = None
        self.sim_mat = None
        self.pred_rate = None
        self.I = None #user_size
        self.U = None #item_size
        self.rmse_error = 0
        self.mae_error = 0

    def calc_similarity(self):
        self.U = self.train_mat.shape[0] # 406
        self.I = self.train_mat.shape[1] # 31
        self.sim_mat = np.zeros((self.U, self.U))
        print(self.train_mat.shape)
        print(self.sim_mat.shape)
        for i in range(0, self.U):
            for j in range(0, self.U):
                if i == j:
                    self.sim_mat[i][j] = self.sim_mat[j][i] = 0
                else :
                    self.sim_mat[i][j] = self.sim_mat[j][i] = cossim(self.train_mat[i], self.train_mat[j])
                    #self.sim_mat[i][j] = self.sim_mat[j[i]]
        print("rate matrix calculation is safely over")

    def train(self, train_df, valid_df):
        """
        expected attribute:
        pd.DataFrameObject
        (cols = [overall, reviewerID, asin])

        return : none

        """
        self.train_mat = self.to_matrix(train_df).values
        self.valid_mat = self.to_matrix(valid_df).values
        self.calc_similarity()
        #self.predict()

    def predict(self, test_df):
        self.test_mat = self.to_matrix(test_df).values

        self.pred_mat = np.zeros(self.train_mat.shape)
        for u in range(0, self.U):
            simsum = np.sum(np.abs(self.sim_mat[u]))
            for i in range(0, self.I):
                rate = self.train_mat[:, i]
                self.pred_mat[u][i] = np.dot(self.sim_mat[u], rate) /simsum
        print("prediction is safely over")


    def evaluate(self):
        e1, e2 = self.evaluate_(self.test_mat, self.pred_mat)
        print("test: rmse is ", e1, ", mae is ", e2)







