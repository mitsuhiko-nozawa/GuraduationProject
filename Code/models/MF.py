from .trainer import trainer
from utils.util import cossim, rmse, mae

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

        # アイテムごと、ユーザーごとにまとめる
        self.items_by_user, self.users_by_item = self.make_data(train);

        # data preparation
        # user feature and item feature
        self.U = np.random.rand(self.user_size, dim_size)
        self.I = np.random.rand(self.item_size, dim_size)

        # all mean, user mean, item mean
        self.m = np.sum(train["overall"]) / self.train_size

        self.bu = np.zeros(self.user_size)
        for i in range(self.user_size):
            item_dict = self.items_by_user[i]
            self.bu[i] = sum(item_dict.values()) / len(item_dict)
            if self.bu[i] != self.bu[i]:
                print("nan is inspected from user ", i)
                self.bu[i] = 0

        self.bi = np.zeros(self.item_size)
        for i in range(self.item_size):
            user_dict = self.users_by_item[i]
            self.bi[i] = sum(user_dict.values()) / len(user_dict)
            if self.bi[i] != self.bi[i]:
                print("nan is inspected from item ", i)
                self.bi[i] = 0

        self.bi.reshape(-1, self.item_size)


        # train
        for epoch in range(epoch_size):
            tempU = self.U.copy()
            tempI = self.I.copy()
            train_losses = []


            for i, j, overall in zip(train["reviewerID"], train["asin"], train["overall"]):

                pred = np.dot(self.U[i],  self.I[j]) + self.m + self.bu[i] + self.bi[j]
                # trainに存在するものだけ更新
                error = overall - pred
                train_losses.append(error)
                tempU[i] = tempU[i] + lr * ( error * tempI[j] - r * tempU[i] )
                tempI[j] = tempI[j] + lr * ( error * tempU[i] - r * tempI[j] )

                # tempU[i] += error * self.I[j]
                # tempI[j] += error * self.U[i]

            self.U = tempU
            self.I = tempI
            # self.U = self.U + lr * (tempU - r * self.U)
            # self.I = self.I + lr * (tempI - r * self.I)
            self.train_losses.append(rmse(train_losses))


            # validation
            valid_losses = []
            for i, j, overall in zip( valid["reviewerID"], valid["asin"], valid["overall"] ):
                pred = np.dot(self.U[i], self.I[j]) + self.m + self.bu[i] + self.bi[j]
                valid_losses.append(pred - overall)
                # print(pred, overall)
            self.valid_losses.append(rmse(valid_losses))

            # nan の検出
            if self.train_losses[-1] != self.train_losses[-1] or self.valid_losses[-1] != self.valid_losses[-1]:
                print("overflow is occuring")
                print("stop training")
                return 0

            # describe
            if (epoch+1) % desc_step == 0:
                print("epoch ", epoch+1, " is done")
                train_loss = self.train_losses[-1]
                valid_loss = self.valid_losses[-1]
                print("train loss is : ", train_loss, ", valid loss is :", valid_loss)


            # early stopping and epoch ending
            if (epoch != 0 and early_stopping and self.valid_losses[-2] < self.valid_losses[-1]) or (epoch+1) == epoch_size:
                print("epoch ", epoch+1, " is done")
                train_loss = self.train_losses[-1]
                valid_loss = self.valid_losses[-1]
                print("train loss is : ", train_loss, ", valid loss is :", valid_loss)
                """
                path = "models/losses/MF/loss_hp.hdf5"
                try:
                    with h5py.File(path, 'w') as f:
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
                """
                return 0


    def predict(self, test):
        overalls = []
        losses = []
        for i, j, overall in zip(test["reviewerID"], test["asin"], test["overall"]):
            pred = np.dot(self.U[i], self.I[j]) + self.m + self.bu[i] + self.bi[j]
            overalls.append(pred)
            losses.append(pred - overall)
            # print(pred, overall)
        self.preds["reviewerID"] = test["reviewerID"]
        self.preds["asin"] = test["asin"]
        self.preds["overall"] = np.array(overalls)
        self.test_losses = losses

    def evaluate(self):
        loss = rmse(self.test_losses)
        print("test loss is : ", loss)










