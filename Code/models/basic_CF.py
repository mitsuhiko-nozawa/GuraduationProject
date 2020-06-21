import numpy as np
import pandas as pd
from .trainer import trainer
from utils.util import cossim
np.set_printoptions(precision=3)


class cf_trainer(trainer):
    def __init__(self, user_dict, item_dict):
        super().__init__(user_dict, item_dict)
        """
        親クラスのコンストラクタ
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.user_size = len(self.user_dict)
        self.item_size = len(self.item_dict)
        self.items_by_user = None
        self.users_by_item = None

        self.train_size = None
        self.test_size = None

        self.errors = []
        """


    def train(self, train):
        self.train_size = len(train["asin"])

        # ユーザごとのアイテム、アイテムごとのユーザをまとめる
        self.items_by_user, self.users_by_item = self.make_data(train)



    def predict(self, test):
        self.test_size = len(test["asin"])

        for user_id, item_id, overall in zip(test["reviewerID"], test["asin"], test["overall"]):
            sim_vec = self.calc_sim(user_id)
            item_vec = self.coo2vec(self.users_by_item[item_id], self.user_size)
            if np.sum(np.abs(sim_vec)) == 0:
                # おそらく共通点が全くない
                print("invalid data is ", user_id, " ", item_id, " ", overall)
                # print(sim_vec)
                continue
            pred_overall = np.dot(sim_vec, item_vec) / np.sum(np.abs(sim_vec))
            self.errors.append(pred_overall - overall)
        self.errors = np.array(self.errors)

    def calc_sim(self, userId): #ユーザとそれ以外の人との類似度を計算する
        user_vec = self.coo2vec(self.items_by_user[userId], self.item_size)
        sim_vec = np.zeros(self.user_size)
        for i in range(self.user_size):
            if i == userId:
                sim_vec[i] = 0.0;
            else:
                other_vec = self.coo2vec(self.items_by_user[i], self.item_size)
                sim_vec[i] = cossim(user_vec, other_vec)
        return sim_vec




    def evaluate(self):
        rmse = (np.sum(self.errors**2)/self.errors.shape[0])**0.5
        mae = np.sum(np.abs(self.errors)) / self.errors.shape[0]
        print("test: rmse is ", rmse, ", mae is ", mae)







