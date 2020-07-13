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


        # ALS trainを {userid : ([itemid, itemid, ...], [overall, overall, ...]),  ...} の形にする

        ALS_train_user = {}
        for userid, itemid, overall in zip(train["reviewerID"], train["asin"], train["overall"]):
            if userid not in ALS_train_user:
                ALS_train_user[userid] = ([], [])
            ALS_train_user[userid][0].append(itemid)
            ALS_train_user[userid][1].append(overall)

        ALS_train_item = {}
        for userid, itemid, overall in zip(train["reviewerID"], train["asin"], train["overall"]):
            if itemid not in ALS_train_item:
                ALS_train_item[itemid] = ([], [])
            ALS_train_item[itemid][0].append(userid)
            ALS_train_item[itemid][1].append(overall)

        del train

        for epoch in range(epoch_size):
            train_losses = []
            for i, items in ALS_train_user.items():  # ユーザをfixしてアイテムのみ学習
                idx = items[0]
                overalls = np.array(items[1])
                preds = np.sum(self.U[i] * self.I[idx], axis=1) + self.m + self.bu[i] + self.bi[idx]
                errors = (overalls - preds)
                train_losses += errors.tolist()
                self.I[idx] = self.I[idx] + lr * (np.matmul(errors.reshape(-1, 1), self.U[i].reshape(1, -1)) - r * self.I[idx])
                self.bi[idx] = self.bi[idx] + lr * (errors - r * self.bi[idx])


            for j, users in ALS_train_item.items():  # アイテムをfixしてユーザのみ学習
                idx = users[0]
                overalls = np.array(users[1])
                preds = np.sum(self.I[j] * self.U[idx], axis=1) + self.m + self.bu[idx] + self.bi[j]
                errors = (overalls - preds)
                train_losses += errors.tolist()
                self.U[idx] = self.U[idx] + lr * (np.matmul(errors.reshape(-1, 1), self.I[j].reshape(1, -1)) - r * self.U[idx])
                self.bu[idx] = self.bu[idx] + lr * (errors - r * self.bu[idx])

            self.train_losses.append(mse(train_losses))

            # validation
            valid_losses = []
            for i, j, overall in zip(valid["reviewerID"], valid["asin"], valid["overall"]):
                pred = np.dot(self.U[i], self.I[j]) + self.m + self.bu[i] + self.bi[j]
                valid_losses.append(pred - overall)

            self.valid_losses.append(mse(valid_losses))
            if self.valid_losses[-1] != self.valid_losses[-1]:
                print("epoch ",epoch+1,  " Invalid : overflow or zero division is occuring")
                return 0

            # describe
            if (epoch+1) % desc_step == 0:
                self.desc(epoch)


            # early stopping
            if epoch != 0 and self.valid_losses[-2] < self.valid_losses[-1] and early_stopping:
                self.desc(epoch)
                return 0
        self.desc(epoch_size-1)



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
        return loss


    def desc(self, epoch):
        print("epoch ", epoch + 1, " is done")
        print("train loss is : ", self.train_losses[-1], ", valid loss is :", self.valid_losses[-1])









