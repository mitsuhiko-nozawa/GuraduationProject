import numpy as np
import pandas as pd
from utils.util import rmse, mae

class trainer():
    def __init__(self, user_dict, item_dict):
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.user_size = len(self.user_dict)
        self.item_size = len(self.item_dict)
        self.items_by_user = None
        self.users_by_item = None

        self.train_size = None
        self.valid_size = None
        self.test_size = None

        self.preds = {}  # { "reviewerID" : [], "asin" : [], "overall" : []}

        self.errors = []


    def make_data(self, train):
        items_by_user = {}
        users_by_item = {}
        for i in range(self.train_size):
            user_id = train["reviewerID"][i]
            item_id = train["asin"][i]
            overall = train["overall"][i]
            if user_id not in items_by_user:
                items_by_user[user_id] = {}
            items_by_user[user_id][item_id] = overall

            if item_id not in users_by_item:
                users_by_item[item_id] = {}
            users_by_item[item_id][user_id] = overall


        return items_by_user, users_by_item


    def coo2vec(self, coo, vec_size):  # coo = {id1 : overall, id2 : overall, ...}
        res = np.zeros(vec_size)
        for key, val in coo.items():
            res[key] = val
        return res






