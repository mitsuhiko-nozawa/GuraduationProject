from .trainer import trainer
from utils.util import cossim, mse

import numpy as np
# import h5py
np.random.seed(0)

class mf_trainer(trainer):
    def __init__(self, user_dict, item_dict):
        super().__init__(user_dict, item_dict)
        """
        親クラス内のコンストラクタ
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.user_size = len(self.user_dict)
        self.item_size = len(self.item_dict)

        self.items_by_user = None
        self.users_by_item = None

        self.errors = []

        """

        self.U = None  # (U_size , dim_size)
        self.I = None  # (I_size, dim_size)
        self.m = None  # 全体の平均
        self.bu = None  # アイテムの平均
        self.bi = None  # ユーザーの平均

        self.train_losses = []
        self.valid_losses = []
        self.test_losses = []



    def train(self, train, valid, dim_size, r, lr, epoch_size, desc_step, early_stopping):
        # param config
        self.train_size = len(train["asin"])
        self.valid_size = len(valid["asin"])

        # data preparation
        # user feature and item feature
        self.U = np.random.rand(self.user_size, dim_size)
        self.I = np.random.rand(self.item_size, dim_size)

        # all average, bias of user, bias of item
        self.m = np.sum(train["overall"]) / self.train_size
        self.bu = np.random.rand(self.user_size)
        self.bi = np.random.rand(self.item_size)


        # ALS trainを {userid : [(itemid, overall), ...], userid : [(itemid, overall), ...] , ...} の形にする
        ALS_train_user = {}
        for userid, itemid, overall in zip(train["reviewerID"], train["asin"], train["overall"]):
            if userid not in ALS_train_user:
                ALS_train_user[userid] = []
            ALS_train_user[userid].append( (itemid, overall) )

        ALS_train_item = {}
        for userid, itemid, overall in zip(train["reviewerID"], train["asin"], train["overall"]):
            if itemid not in ALS_train_item:
                ALS_train_item[itemid] = []
            ALS_train_item[itemid].append( (userid, overall) )
        del train


        for epoch in range(epoch_size):
            train_losses = []
            for i, items in ALS_train_user.items():  # ユーザをfixしてアイテムのみ学習
                for t in items:
                    j = t[0]
                    overall = t[1]
                    # pred = np.sum(selfU[i] * self.I[idx], axis=1) +
                    pred = np.dot(self.U[i], self.I[j]) + self.m + self.bu[i] + self.bi[j]
                    error = overall - pred
                    train_losses.append(error)
                    self.I[j] = self.I[j] + lr * (error * self.U[i] - r * self.I[j])
                    self.bi[j] = self.bi[j] + lr * (error - r * self.bi[j])
            for j, users in ALS_train_item.items():  # アイテムをfixしてユーザのみ学習
                for t in users:
                    i = t[0]
                    overall = t[1]
                    pred = np.dot(self.U[i], self.I[j]) + self.m + self.bu[i] + self.bi[j]
                    error = overall - pred
                    train_losses.append(error)
                    self.U[i] = self.U[i] + lr * (error * self.I[j] - r * self.U[i])
                    self.bu[i] = self.bu[i] + lr * (error - r * self.bu[i])
            self.train_losses.append(mse(train_losses))

            # validation
            valid_losses = []
            for i, j, overall in zip( valid["reviewerID"], valid["asin"], valid["overall"] ):
                pred = np.dot(self.U[i], self.I[j]) + self.m + self.bu[i] + self.bi[j]
                valid_losses.append(pred - overall)

            self.valid_losses.append(mse(valid_losses))
            if self.valid_losses[-1] != self.valid_losses[-1]:
                print("Invalid : overflow or zero division is occuring")
                return 0

            # describe
            if (epoch+1) % desc_step == 0:
                print("epoch ", epoch+1, " is done")
                train_loss = self.train_losses[-1]
                valid_loss = self.valid_losses[-1]
                print("train loss is : ", train_loss, ", valid loss is :", valid_loss)


            # early stopping or epoch ending
            if (epoch != 0 and early_stopping and self.valid_losses[-2] < self.valid_losses[-1]) or (epoch+1) == epoch_size:
                print("epoch ", epoch+1, " is done")
                train_loss = self.train_losses[-1]
                valid_loss = self.valid_losses[-1]
                print("train loss is : ", train_loss, ", valid loss is :", valid_loss)
                return 0


    def predict(self, test):
        overalls = []
        losses = []
        for i, j, overall in zip(test["reviewerID"], test["asin"], test["overall"]):
            pred = np.dot(self.U[i], self.I[j]) + self.m + self.bu[i] + self.bi[j]
            overalls.append(pred)
            losses.append(pred - overall)

        self.preds["reviewerID"] = test["reviewerID"]
        self.preds["asin"] = test["asin"]
        self.preds["overall"] = np.array(overalls)
        self.test_losses = losses

    def evaluate(self):
        loss = mse(self.test_losses)
        print("     test loss is : ", loss)












