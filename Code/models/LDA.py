from .trainer import trainer
from utils.util import cossim, mse, mae
from gensim import corpora
from gensim.models.ldamodel import LdaModel

import numpy as np
np.random.seed(0)

class LDA_trainer(trainer):
    def __init__(self, user_dict, item_dict, words_by_item):
        super().__init__(user_dict, item_dict)
        self.words_by_item = words_by_item
        self.U = None
        self.I = None

        self.m = None  # all rating average
        self.bu = None  # bias of user
        self.bi = None  # bias of item

        self.train_losses = []
        self.valid_losses = []
        self.test_losses = []

    def train(self, train, valid, dim_size, r, lr, epoch_size, desc_step, early_stopping):
        # LDA Part
        words_list = [val for val in self.words_by_item.values()]
        words_dictionary = corpora.Dictionary(words_list)
        corpus = [words_dictionary.doc2bow(words) for words in words_list]
        lda_model = LdaModel(
            corpus=corpus,
            num_topics=dim_size,
            id2word=words_dictionary,
            random_state=1
        )
        self.I = np.zeros((self.item_size, dim_size))
        for id, text in self.words_by_item.items():
            bow = words_dictionary.doc2bow(text)
            topics = lda_model[bow]
            temp = np.zeros(dim_size)
            for t in topics:
                temp[t[0]] = t[1]
            self.I[id] = temp
        ######################################################################

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
            ALS_train_user[userid].append((itemid, overall))

        ALS_train_item = {}
        for userid, itemid, overall in zip(train["reviewerID"], train["asin"], train["overall"]):
            if itemid not in ALS_train_item:
                ALS_train_item[itemid] = []
            ALS_train_item[itemid].append((userid, overall))
        del train

        for epoch in range(epoch_size):
            train_losses = []
            for i, items in ALS_train_user.items():  # ユーザをfixしてアイテムのみ学習
                for t in items:
                    j = t[0]
                    overall = t[1]
                    pred = np.dot(self.U[i], self.I[j]) + self.m + self.bu[i] + self.bi[j]
                    error = overall - pred
                    train_losses.append(error)
                    # self.I[j] = self.I[j] + lr * (error * self.U[i] - r * self.I[j])
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
            for i, j, overall in zip(valid["reviewerID"], valid["asin"], valid["overall"]):
                pred = np.dot(self.U[i], self.I[j]) + self.m + self.bu[i] + self.bi[j]
                valid_losses.append(pred - overall)
            self.valid_losses.append(mse(valid_losses))
            if self.valid_losses[-1] != self.valid_losses[-1]:
                print("invalid")
                return 0

            # describe
            if (epoch + 1) % desc_step == 0:
                print("epoch ", epoch + 1, " is done")
                train_loss = self.train_losses[-1]
                valid_loss = self.valid_losses[-1]
                print("train loss is : ", train_loss, ", valid loss is :", valid_loss)

            # early stopping or epoch ending
            if (epoch != 0 and early_stopping and self.valid_losses[-2] < self.valid_losses[-1]) or (
                    epoch + 1) == epoch_size:
                print("epoch ", epoch + 1, " is done")
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






