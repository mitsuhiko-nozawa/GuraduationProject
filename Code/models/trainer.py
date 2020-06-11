import numpy as np
import pandas as pd
from utils.util import rmse, mae

class trainer():
    def __init__(self):
        self.param = 1

    def to_matrix(self, df):
        if "reviewText" in df.columns:
            df.drop("reviewText", axis=1, inplace=True)
        df = df.pivot_table(values=["overall"], index=["reviewerID"], columns=["asin"])
        df.fillna(0, inplace=True)
        return df

    def evaluate_(self, label_mat, pred_mat):
        rmse_error = rmse(label_mat, pred_mat)
        mae_error = mae(label_mat, pred_mat)
        #print("rmse is : ", rmse_error)
        #print("mae is : ", mae_error)
        return rmse_error, mae_error



