from .trainer import trainer
from utils.util import rmse, mae

import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Dot
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D



class DeepCoNN_trainer(trainer):
    def __init__(self, user_dict, item_dict,
                 train_text_by_user, train_text_by_item,
                 valid_text_by_user, valid_text_by_item,
                 test_text_by_user, test_text_by_item,
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
        self.train_text_by_user = train_text_by_user
        self.train_text_by_item = train_text_by_item
        self.valid_text_by_user = valid_text_by_user
        self.valid_text_by_item = valid_text_by_item
        self.test_text_by_user = test_text_by_user
        self.test_text_by_item = test_text_by_item


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
        for userid, itemid in zip(train["reviewerID"], train["asin"]):
            user_texts.append(self.pad(self.train_text_by_user[userid]))
            item_texts.append(self.pad(self.train_text_by_item[itemid]))


        train_X = [np.array(user_texts), np.array(item_texts)]
        train_Y = train["overall"]

        # valid
        user_texts = []
        item_texts = []
        for userid, itemid in zip(valid["reviewerID"], valid["asin"]):
            user_texts.append(self.pad(self.valid_text_by_user[userid]))
            item_texts.append(self.pad(self.valid_text_by_item[itemid]))

        valid_X = [np.array(user_texts), np.array(item_texts)]
        valid_Y = valid["overall"]

        del user_texts, item_texts

        # training
        print("==========training==========")
        self.model.fit(train_X, train_Y, batch_size=self.batch_size, epochs=self.epoch, validation_data=(valid_X, valid_Y))
        # self.model.save_weights('param.hdf5')







    def predict(self, test):
        user_texts = []
        item_texts = []
        for userid, itemid in zip(test["reviewerID"], test["asin"]):
            user_texts.append(self.pad(self.test_text_by_user[userid]))
            item_texts.append(self.pad(self.test_text_by_item[itemid]))

        test_X = [np.array(user_texts), np.array(item_texts)]
        test_Y = test["overall"]
        preds = self.model.predict(test_X)
        self.errors = preds-test_Y



    def evaluate(self):
        loss = rmse(self.errors)
        print("test loss is : ", loss)


    def pad(self, text):
        # input :
        # print(text.shape)
        # print()
        res = np.pad(text, [(0, self.maxlen-text.shape[0]), (0, 0)])
        # print(res.shape)
        return res


    def make_model(self):

        input1 = Input(shape=(self.maxlen, self.embedding_dim))
        conv1 = Conv1D(filters=self.filter_num, kernel_size=self.filter_size, padding="valid", activation="relu", strides=1)(input1)
        pool1 = GlobalMaxPooling1D()(conv1)
        flat1 = Flatten()(pool1)
        dense1 = Dense(self.feature_size)(flat1)

        input2 = Input(shape=(self.maxlen, self.embedding_dim))
        conv2 = Conv1D(filters=self.filter_num, kernel_size=self.filter_size, padding="valid", activation="relu", strides=1)(input2)
        pool2 = GlobalMaxPooling1D()(conv2)
        flat2 = Flatten()(pool2)
        dense2 = Dense(self.feature_size)(flat2)

        output = Dot(axes=1)([dense1, dense2])
        self.model = Model(inputs=[input1, input2], outputs=[output])
        self.model.compile(optimizer="RMSprop", loss="mse")