import random, sys, os
import numpy as np
import pandas as pd

sys.path.append('../')
# from models.MF import mf_trainer
from cython_models.MF_cy import mf_trainer

np.random.seed(0)
random.seed(0)

# データはすでに加工済み
# 各ユーザ、アイテムが5回以上評価されているものを抽出
# data preparation

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
for i, file in enumerate(files):
    print(file, " training start")
    train_df = pd.read_csv("data/train/"+file)
    valid_df = pd.read_csv("data/valid/"+file)
    test_df = pd.read_csv("data/test/"+file)


    user_size = len(set(train_df["reviewerID"].tolist() + valid_df["reviewerID"].tolist() + test_df["reviewerID"].tolist()))
    item_size = len(set(train_df["asin"].tolist() + valid_df["asin"].tolist() + test_df["asin"].tolist()))
    train_size = int(train_df.shape[0])
    valid_size = int(valid_df.shape[0])
    test_size = int(test_df.shape[0])





    # dfで渡すと重いのでリストに変換
    # train = { col1 : [], col2 : [], ...., coln : [] }
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


    # 学習部分
    print("mf training")
    grid_fname = "grids/mf_grid/" + file.replace(".csv.gz", "_grid.csv")
    params = pd.read_csv(grid_fname)
    for i, param in params.iterrows():
        if param["val_loss"] != 0 or param["test_loss"] != 0:
            print("iter ", i, " is skipped")
            continue

        r = param["r"]
        lr = param["lr"]
        dim_size = param["dim_size"]

        print("param ...   lr :", lr, ",  r :", r, ", dim_size :", dim_size)
        mf = mf_trainer(user_size, item_size, train_size, valid_size, test_size)
        mf.train(train, valid, dim_size=dim_size, r=r, lr=lr, epoch_size=2500, desc_step=100, early_stopping=True)
        mf.predict(test)
        test_loss = mf.evaluate()
        val_loss = mf.valid_losses[-1]

        if val_loss == val_loss and test_loss == test_loss:
            params.at[i, "val_loss"] = val_loss
            params.at[i, "test_loss"] = test_loss
        else:
            params.at[i, "val_loss"] = -1
            params.at[i, "test_loss"] = -1
        params.to_csv(grid_fname, index=False)
        print("")