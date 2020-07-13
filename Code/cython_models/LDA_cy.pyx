import numpy as np
cimport numpy as np
cimport cython

from gensim import corpora
from gensim.models.ldamodel import LdaModel

np.random.seed(0)


cdef class LDA_trainer:
    # ここで定義したものがクラス内変数となる
    # self.でアクセスできる
    cdef int user_size
    cdef int item_size
    cdef int train_size
    cdef int valid_size
    cdef int test_size

    cdef int dim_size

    cdef double[:, :] U_
    cdef double[:, :] I_
    cdef double m_
    cdef double[:] bu_
    cdef double[:] bi_

    cdef double[:] train_losses
    cdef public double[:] valid_losses
    cdef double[:] test_losses

    cdef double[:] preds

    cdef dict words_by_item


    def __init__(self, user_size, item_size, words_by_item, train_size, valid_size, test_size):
        self.user_size = user_size
        self.item_size = item_size
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size

        self.words_by_item = words_by_item





    cpdef train(self, train, valid, int dim_size, float r, float lr, int epoch_size, int desc_step, early_stopping):
        cdef long[:] train_users = np.array(train["reviewerID"]).astype(np.int)
        cdef long[:] train_items = np.array(train["asin"]).astype(np.int)
        cdef double[:] train_overalls = train["overall"]

        cdef long[:] valid_users = np.array(valid["reviewerID"]).astype(np.int)
        cdef long[:] valid_items = np.array(valid["asin"]).astype(np.int)
        cdef double[:] valid_overalls = valid["overall"]




        # initialize ###########################
        self.dim_size = dim_size
        self.train_size = len(train["asin"])
        self.valid_size = len(valid["asin"])
        self.train_losses = np.zeros(epoch_size)
        self.valid_losses = np.zeros(epoch_size)

        cdef double[:, :] U = np.random.rand(self.user_size, dim_size)
        cdef double[:, :] I = self.lda()
        cdef double m = np.sum(train_overalls) / self.train_size
        cdef double[:] bu = np.random.rand(self.user_size)
        cdef double[:] bi = np.random.rand(self.item_size)
        ###################################





        # variables used in training #########################

        cdef int userid
        cdef int itemid
        cdef double overall
        cdef double pred
        cdef double error
        cdef double[:] train_losses
        cdef double[:] valid_losses


        # train ##############################
        print("train start")
        for epoch in range(epoch_size):
            train_losses = np.zeros(self.train_size)
            for i in range(self.train_size):
                userid = train_users[i]
                itemid = train_items[i]
                overall = train_overalls[i]
                pred = bu[userid] + bi[itemid] + m
                for j in range(dim_size):
                    pred += U[userid, j] * I[itemid, j]
                error = overall - pred
                train_losses[i] = error

                for j in range(dim_size):
                    U[userid, j] = U[userid, j] + lr * ( error * I[itemid, j] - r * U[userid, j] )
                    #I[itemid, j] = I[itemid, j] + lr * ( error * U[userid, j] - r * I[itemid, j] )
                    bu[userid] = bu[userid] + lr * ( error - r * bu[userid] )
                    bi[itemid] = bi[itemid] + lr * ( error - r * bi[itemid] )
                    # print("aaa")

            self.train_losses[epoch] = self.mse(np.asarray(train_losses))


            # valid
            valid_losses = np.zeros(self.valid_size)
            for i in range(self.valid_size):
                userid = valid_users[i]
                itemid = valid_items[i]
                overall = valid_overalls[i]
                pred = bu[userid] + bi[itemid] + m
                for j in range(dim_size):
                    pred += U[userid, j] * I[itemid, j]
                error = overall - pred
                valid_losses[i] = error
            self.valid_losses[epoch] = self.mse(np.asarray(valid_losses))


            if self.valid_losses[epoch] != self.valid_losses[epoch] or self.valid_losses[epoch] < 0.0:
                print("epoch ",epoch+1,  " Invalid : overflow or zero division is occuring")
                self.desc(epoch)
                self.U_ = U
                self.I_ = I
                self.m_ = m
                self.bu_ = bu
                self.bi_ = bi
                return 0

            # describe
            if (epoch+1) % desc_step == 0:
                self.desc(epoch)


            # early stopping
            if epoch != 0 and self.valid_losses[epoch-1] < self.valid_losses[epoch] and early_stopping:
                self.desc(epoch)
                self.U_ = U
                self.I_ = I
                self.m_ = m
                self.bu_ = bu
                self.bi_ = bi
                return 0

        self.desc(epoch_size-1)
        self.U_ = U
        self.I_ = I
        self.m_ = m
        self.bu_ = bu
        self.bi_ = bi

    cpdef predict(self, test):
        cdef long[:] test_users = np.array(test["reviewerID"]).astype(np.int)
        cdef long[:] test_items = np.array(test["asin"]).astype(np.int)
        cdef double[:] test_overalls = test["overall"]


        # initialize ###########################
        self.test_size = len(test["asin"])
        self.test_losses = np.zeros(self.test_size)
        self.preds = np.zeros(self.test_size)
        cdef double[:, :] U = self.U_
        cdef double[:, :] I = self.I_
        cdef double m = self.m_
        cdef double[:] bu = self.bu_
        cdef double[:] bi = self.bi_
        ###################################

        for i in range(self.test_size):
            userid = test_users[i]
            itemid = test_items[i]
            overall = test_overalls[i]
            pred = bu[userid] + bi[itemid] + m
            for j in range(self.dim_size):
                pred += U[userid, j] * I[itemid, j]
            error = overall - pred
            self.test_losses[i] = error
            self.preds[i] = pred



    cpdef evaluate(self):
        loss = self.mse(np.asarray(self.test_losses))
        print("     test loss is : ", loss)
        return loss

    def mse(self, vec):
        vec = np.array(vec)
        return (np.sum(vec ** 2) / vec.shape[0])



    def desc(self, epoch):
        print("epoch ", epoch + 1, " is done")
        print("train loss is : ", self.train_losses[epoch], ", valid loss is :", self.valid_losses[epoch])

    cpdef lda(self):
        words_list = [val for val in self.words_by_item.values()]
        words_dictionary = corpora.Dictionary(words_list)
        corpus = [words_dictionary.doc2bow(words) for words in words_list]
        lda_model = LdaModel(
            corpus=corpus,
            num_topics=self.dim_size,
            id2word=words_dictionary,
            random_state=1
        )
        I = np.zeros((self.item_size, self.dim_size))
        for id, text in self.words_by_item.items():
            bow = words_dictionary.doc2bow(text)
            topics = lda_model[bow]
            temp = np.zeros(self.dim_size)
            for t in topics:
                temp[t[0]] = t[1]
            I[id] = temp
        return I
