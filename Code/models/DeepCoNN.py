from .trainer import trainer
from utils.util import mse, mae

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Dot, Add, Concatenate
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping



class DeepCoNN_trainer:
    def __init__(self, user_size, item_size, parser_,
                 words_by_user_, words_by_item_,
                 embedding_dim, filter_num, filter_size, feature_size, maxlen,
                 batch_size, epoch):
        global parser
        parser = parser_
        global words_by_user
        words_by_user = words_by_user_
        global words_by_item
        words_by_item = words_by_item_

        self.user_size = user_size
        self.item_size = item_size
        self.model = None
        self.maxlen = maxlen

        self.embedding_dim = embedding_dim
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.epoch = epoch



    def train(self, train, valid):
        train_size = len(train["asin"])
        valid_size = len(valid["asin"])

        # create model
        print("==========model creation==========")
        # self.make_model()
        self.make_model2()
        # prepare input
        print("==========generator preparation==========")
        train_generator = Generator(train["reviewerID"], train["asin"], train["overall"], train_size,
                                    self.batch_size, self.maxlen, self.embedding_dim)

        valid_generator = Generator(valid["reviewerID"], valid["asin"], valid["overall"], valid_size,
                                    self.batch_size, self.maxlen, self.embedding_dim)



        # training
        print("==========training==========")
        cb = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
        self.model.fit_generator(
            train_generator,
            epochs=self.epoch,
            #callbacks=[cb],
            validation_data=valid_generator
        )





    def predict(self, test):
        test_size = len(test["asin"])
        test_generator = Generator(test["reviewerID"], test["asin"], test["overall"], test_size,
                                    self.batch_size, self.maxlen, self.embedding_dim)

        self.preds = self.model.predict_generator(
            test_generator
        )

        self.preds = self.preds.reshape(-1)
        self.errors = self.preds-test["overall"]
        print("")


    def evaluate(self):
        loss = mse(self.errors)
        print("test loss is : ", loss)
        return loss


    """
    def make_model(self):

        input1 = Input(shape=(self.maxlen, self.embedding_dim))
        conv1 = Conv1D(filters=self.filter_num, kernel_size=self.filter_size, padding="valid", activation="relu",strides=1)(input1)
        pool1 = GlobalMaxPooling1D()(conv1)
        flat1 = Flatten()(pool1)
        dense1 = Dense(self.feature_size, kernel_regularizer=regularizers.l2(0.001), activation="relu")(flat1)
        dense1 = Dropout(0.2)(dense1, training=True)

        input2 = Input(shape=(self.maxlen, self.embedding_dim))
        conv2 = Conv1D(filters=self.filter_num, kernel_size=self.filter_size, padding="valid", activation="relu", strides=1)(input2)
        pool2 = GlobalMaxPooling1D()(conv2)
        flat2 = Flatten()(pool2)
        dense2 = Dense(self.feature_size, kernel_regularizer=regularizers.l2(0.001), activation="relu")(flat2)
        dense2 = Dropout(0.2)(dense2, training=True)

        output = Dot(axes=1)([dense1, dense2])
        self.model = Model(inputs=[input1, input2], outputs=[output])
        self.model.compile(optimizer=RMSprop(lr=0.002), loss="mse")
    """

    def make_model2(self):
        input1 = Input(shape=(self.maxlen, self.embedding_dim))
        conv1 = Conv1D(filters=self.filter_num,
                       kernel_size=self.filter_size,
                       padding="valid",
                       activation="relu",
                       #kernel_regularizer=regularizers.l2(0.001),
                       #bias_regularizer=regularizers.l2(0.001),
                       strides=1)(input1)
        pool1 = GlobalMaxPooling1D()(conv1)
        flat1 = Flatten()(pool1)
        dense1 = Dense(self.feature_size,
                       #kernel_regularizer=regularizers.l2(0.02),
                       #bias_regularizer=regularizers.l2(0.02),
                       activation="relu")(flat1)
        dense1 = Dropout(0.5)(dense1)


        input2 = Input(shape=(self.maxlen, self.embedding_dim))
        conv2 = Conv1D(filters=self.filter_num,
                       kernel_size=self.filter_size,
                       padding="valid",
                       activation="relu",
                       #kernel_regularizer=regularizers.l2(0.001),
                       #bias_regularizer=regularizers.l2(0.001),
                       strides=1)(input2)
        pool2 = GlobalMaxPooling1D()(conv2)
        flat2 = Flatten()(pool2)
        dense2 = Dense(self.feature_size,
                       #kernel_regularizer=regularizers.l2(0.02),
                       #bias_regularizer=regularizers.l2(0.02),
                       activation="relu")(flat2)
        dense2 = Dropout(0.5)(dense2)


        output2 = Concatenate()([dense1, dense2])
        #output2 = Dense(1,
        #                kernel_regularizer=regularizers.l2(0.001),
        #                bias_regularizer=regularizers.l2(0.001),
        #                activation="relu"
        #                )(output2)
        output2 = Dense(1)(output2)

        self.model = Model(inputs=[input1, input2], outputs=[output2])
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss="mse")
        #self.model.compile(optimizer=RMSprop(lr=0.000015), loss="mse")


class Generator(Sequence):
    def __init__(self, users, items, overalls, data_size,
                 batch_size, maxlen, embedding_dim):
        self.users = users
        self.items = items
        self.overalls = overalls
        self.data_size = data_size
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim

    def __getitem__(self, idx):
        # バッチの番号
        #X, yを返す

        users = self.users[idx*self.batch_size:(idx+1)*self.batch_size]
        items = self.items[idx*self.batch_size:(idx+1)*self.batch_size]
        size = len(users)

        user_tensor = np.zeros((size, self.maxlen, self.embedding_dim))
        item_tensor = np.zeros((size, self.maxlen, self.embedding_dim))

        # make tensor
        for i, userid in enumerate(users):
            words = words_by_user[userid]
            mat = np.zeros((self.maxlen, self.embedding_dim))
            for j, word in enumerate(words):
                vec = parser[word][:self.embedding_dim]
                mat[j] = vec
            user_tensor[i] = mat

        for i, itemid in enumerate(items):
            words = words_by_item[itemid]
            mat = np.zeros((self.maxlen, self.embedding_dim))
            for j, word in enumerate(words):
                vec = parser[word][:self.embedding_dim]
                mat[j] = vec
            item_tensor[i] = mat

        X = [user_tensor, item_tensor]
        y = self.overalls[idx*self.batch_size:(idx+1)*self.batch_size]

        return X, y

    def __len__(self):
        # バッチ数を返す
        return int((self.data_size-1)/self.batch_size+1)

    def on_eopch_end(self):
        pass
