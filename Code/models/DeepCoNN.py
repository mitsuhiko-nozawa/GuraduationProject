from .trainer import trainer
from utils.util import mse, mae

import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Dot
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import RMSprop



class DeepCoNN_trainer(trainer):
    def __init__(self, user_dict, item_dict,
                 text_by_user, text_by_item,
                 embedding_dim, filter_num, filter_size, feature_size, maxlen,
                 batch_size, epoch):
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
        self.text_by_user = text_by_user
        self.text_by_item = text_by_item


        self.model = None
        self.maxlen = maxlen

        self.embedding_dim = embedding_dim
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.epoch = epoch

    def train(self, train, valid):

        # create model
        print("==========model creation==========")
        self.make_model()
        # prepare input
        print("==========data preparation==========")
        #train
        user_texts = []
        item_texts = []
        overalls = []
        for userid, itemid, overall in zip(train["reviewerID"], train["asin"], train["overall"]):
            user_texts.append(self.text_by_user[userid])
            item_texts.append(self.text_by_item[itemid])
            overalls.append(overall)


        train_X = [np.array(user_texts), np.array(item_texts)]
        train_Y = np.array(overalls)
        del user_texts, item_texts, overalls, train

        # valid
        user_texts = []
        item_texts = []
        overalls = []
        for userid, itemid, overall in zip(valid["reviewerID"], valid["asin"], valid["overall"]):
            user_texts.append(self.text_by_user[userid])
            item_texts.append(self.text_by_item[itemid])
            overalls.append(overall)

        valid_X = [np.array(user_texts), np.array(item_texts)]
        valid_Y = np.array(overalls)

        del user_texts, item_texts, overalls

        # training
        print("==========training==========")
        self.model.fit(train_X, train_Y, batch_size=self.batch_size, epochs=self.epoch, validation_data=(valid_X, valid_Y))
        # self.model.save_weights('param.hdf5')




    def predict(self, test):
        user_texts = []
        item_texts = []
        overalls = []
        for userid, itemid, overall in zip(test["reviewerID"], test["asin"], test["overall"]):
            user_texts.append(self.text_by_user[userid])
            item_texts.append(self.text_by_item[itemid])
            overalls.append(overall)


        test_X = [np.array(user_texts), np.array(item_texts)]
        test_Y = np.array(overalls)
        del user_texts, item_texts, overalls, test

        self.preds = self.model.predict(test_X).reshape(-1)
        self.errors = self.preds-test_Y
        print("")


    def evaluate(self):
        loss = mse(self.errors)
        print("test loss is : ", loss)


    def make_model(self):

        input1 = Input(shape=(self.maxlen, self.embedding_dim))
        conv1 = Conv1D(filters=self.filter_num, kernel_size=self.filter_size, padding="valid", activation="relu",strides=1)(input1)
        pool1 = GlobalMaxPooling1D()(conv1)
        flat1 = Flatten()(pool1)
        dense1 = Dense(self.feature_size)(flat1)
        dense1 = Dropout(0.2)(dense1, training=True)

        input2 = Input(shape=(self.maxlen, self.embedding_dim))
        conv2 = Conv1D(filters=self.filter_num, kernel_size=self.filter_size, padding="valid", activation="relu", strides=1)(input2)
        pool2 = GlobalMaxPooling1D()(conv2)
        flat2 = Flatten()(pool2)
        dense2 = Dense(self.feature_size)(flat2)
        dense2 = Dropout(0.2)(dense2, training=True)

        output = Dot(axes=1)([dense1, dense2])
        self.model = Model(inputs=[input1, input2], outputs=[output])
        self.model.compile(optimizer=RMSprop(lr=0.002), loss="mse")

