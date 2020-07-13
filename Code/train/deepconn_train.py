import random, sys, os
sys.path.append('../')
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from utils.parser import parse, parse2, parse3

from models.DeepCoNN import DeepCoNN_trainer

np.random.seed(0)
random.seed(0)


files = [
    "AMAZON_FASHION__1.csv.gz",
    "AMAZON_FASHION__5.csv.gz",
    "AMAZON_FASHION__10.csv.gz",
    "yelp_academic_dataset_review__1.csv.gz",
    "yelp_academic_dataset_review__5.csv.gz",
    "yelp_academic_dataset_review__10.csv.gz",
    "bgg-13m-reviews__1.csv.gz",
    "bgg-13m-reviews__5.csv.gz",
    "bgg-13m-reviews__10.csv.gz"
]

path = "../../Dataset/processed/"
path2 = "../../Dataset/Corpus/GoogleNews-vectors-negative300.bin.gz"
parser = KeyedVectors.load_word2vec_format(path2, binary=True)
print("parser is builded")
MAXLEN = 300
# MAXLEN = 100

for i, file in enumerate(files):
    print(file, " training start")

    train_df = pd.read_csv("data/train/"+file)
    valid_df = pd.read_csv("data/valid/"+file)
    test_df = pd.read_csv("data/test/"+file)
    words_by_user_df = pd.read_csv("data/words_by_user/" + file)
    words_by_item_df = pd.read_csv("data/words_by_item/" + file)

    user_size = len(set(train_df["reviewerID"].tolist() + valid_df["reviewerID"].to_list() + test_df["reviewerID"].to_list()))
    item_size = len(set(train_df["asin"].tolist() + valid_df["asin"].to_list() + test_df["asin"].to_list()))
    train_size = int(train_df.shape[0])
    valid_size = int(valid_df.shape[0])
    test_size = int(test_df.shape[0])


    #words_by_user_df["words"] = words_by_user_df["words"].apply(
    #    lambda x: [ y[1:-1] for y in x.replace("[", "").replace("]", "").replace(",", "").split()][:MAXLEN])

    #words_by_item_df["words"] = words_by_item_df["words"].apply(
    #    lambda x: [ y[1:-1] for y in x.replace("[", "").replace("]", "").replace(",", "").split()][:MAXLEN])


    words_by_user = dict(zip(words_by_user_df["user_id"], words_by_user_df["words"]))
    words_by_item = dict(zip(words_by_item_df["item_id"], words_by_item_df["words"]))
    del words_by_item_df, words_by_user_df

    for key, val in words_by_user.items():
        words_by_user[key] = [x[1:-1] for x in val.replace("[", "").replace("\\", "").replace("]", "").replace(",", "").split()][:MAXLEN]
    for key, val in words_by_item.items():
        words_by_item[key] = [x[1:-1] for x in val.replace("[", "").replace("\\", "").replace("]", "").replace(",", "").split()][:MAXLEN]



    # train = { reviewerID : [...], asin : [...], overall : [...] }
    train = {}
    valid = {}
    test = {}
    # train
    train["reviewerID"] = train_df["reviewerID"]
    train["asin"] = train_df["asin"]
    train["overall"] = train_df["overall"].values
    # valid
    valid["reviewerID"] = valid_df["reviewerID"]
    valid["asin"] = valid_df["asin"]
    valid["overall"] = valid_df["overall"].values
    # test
    test["reviewerID"] = test_df["reviewerID"]
    test["asin"] = test_df["asin"]
    test["overall"] = test_df["overall"].values
    del train_df, valid_df, test_df

    """
    MAXLEN = 0
    for words in words_by_user.values():
        MAXLEN = max(MAXLEN, len(words))
    for words in words_by_item.values():
        MAXLEN = max(MAXLEN, len(words))
    print("MAXLEN is ", MAXLEN)
    
    
    MAXLEN = 100
    for key, val in words_by_user.items():
        words_by_user[key] = val[:MAXLEN]
    for key, val in words_by_item.items():
        words_by_item[key] = val[:MAXLEN]
    """


    # 学習部分
    print("")
    print("DeepCoNN training")
    grid_fname = "grids/deepconn_grid/" + file.replace(".csv.gz", "_grid.csv")
    params = pd.read_csv(grid_fname)
    for i, param in params.iterrows():
        if param["val_loss"] != 0 or param["test_loss"] != 0:
            print("iter ", i, " is skipped")
            #continue
        embedding_dim = int(param["embedding_dim"])
        filter_size = int(param["filter_size"])
        # lr = param["lr"]
        batch_size = int(param["batch_size"])

        # filter_size = 3
        # embedding_dim = 25
        # batch_size = 100
        print("param ...   lr :", lr)
        DeepCoNN = DeepCoNN_trainer(user_size, item_size, parser,
                                    words_by_user, words_by_item,
                                    embedding_dim=embedding_dim, filter_num=100, filter_size=filter_size,
                                    feature_size=25, maxlen=MAXLEN,
                                    batch_size=batch_size, epoch=1)
        DeepCoNN.train(train, valid)
        DeepCoNN.predict(test)
        test_loss = DeepCoNN.evaluate()
        val_loss = DeepCoNN = -1

        if val_loss == val_loss and test_loss == test_loss:
            params.at[i, "val_loss"] = val_loss
            params.at[i, "test_loss"] = test_loss
        else:
            params.at[i, "val_loss"] = -1
            params.at[i, "test_loss"] = -1
        params.to_csv(grid_fname, index=False)
        print("")
