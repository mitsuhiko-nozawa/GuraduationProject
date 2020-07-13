import random, sys, os
sys.path.append('../')

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from utils.parser import parse3

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


path2 = "../../Dataset/Corpus/GoogleNews-vectors-negative300.bin.gz"
parser = KeyedVectors.load_word2vec_format(path2, binary=True)
print("parser is builded")

for i, file in enumerate(files):
    path = "../../Dataset/processed/"
    df = pd.read_csv(path+file)
    print(file, " is successfuly loaded")
    print("df shape is ", df.shape)
    data_size = df.shape[0]


    # train, valid, testにスプリット
    # validとtestのインデックスを抽出
    valid_size = test_size = int(data_size*0.15)
    val_tes_index = random.sample(range(data_size), valid_size+test_size)
    val_index = random.sample(val_tes_index, valid_size)
    test_index = list(set(val_tes_index) - set(val_index))

    valid_df = df.iloc[val_index]
    test_df = df.iloc[test_index]
    train_df = df.drop(index=val_tes_index)
    del val_tes_index, df, val_index, test_index, valid_size, data_size


    # trainのテキストのみparseする
    # words_by_item = { itemid1 : words, itemid2 : words, ...}
    words_by_item, words_by_user, notParsedWords, drop_user, drop_item = parse3(parser, train_df)

    # trainで一件もテキストを持たない人を削除
    # valid, test にいてtrainでテキストを持たないデータを削除
    for userid in list(set(train_df["reviewerID"].unique().tolist() +
                           valid_df["reviewerID"].unique().tolist() +
                           test_df["reviewerID"].unique().tolist() )):
        try:
            if len(words_by_user[userid]) == 0:
                drop_user.append(userid)
                #print("oops user")
        except:
            drop_user.append(userid)


    for itemid in list(set(train_df["asin"].unique().tolist() +
                           valid_df["asin"].unique().tolist() +
                           test_df["asin"].unique().tolist() )):
        try:
            if len(words_by_item[itemid]) == 0:
                drop_item.append(itemid)
                #print("oops item")
        except:
            drop_item.append(itemid)



    print("user num not have text is ", len(drop_user))
    print("item num not have text is ", len(drop_item))
    train_df.drop(index=train_df[train_df["reviewerID"].isin(drop_user)].index, inplace=True)
    train_df.drop(index=train_df[train_df["asin"].isin(drop_item)].index, inplace=True)
    valid_df.drop(index=valid_df[valid_df["reviewerID"].isin(drop_user)].index, inplace=True)
    valid_df.drop(index=valid_df[valid_df["asin"].isin(drop_item)].index, inplace=True)
    test_df.drop(index=test_df[test_df["reviewerID"].isin(drop_user)].index, inplace=True)
    test_df.drop(index=test_df[test_df["asin"].isin(drop_item)].index, inplace=True)
    del drop_user, drop_item


    # user_dictとitem_dictのインデックスを振り直し
    user_dict = {}
    item_dict = {}
    users = list(set(train_df["reviewerID"].unique().tolist() +
                     valid_df["reviewerID"].unique().tolist() + test_df["reviewerID"].unique().tolist()))
    items = list(set(train_df["asin"].unique().tolist() +
                     valid_df["asin"].unique().tolist() + test_df["asin"].unique().tolist()))

    for i, id in enumerate(users):
        user_dict[id] = i
    for i, id in enumerate(items):
        item_dict[id] = i

    # user_dict, item_dictにいるデータだけを抽出
    words_by_item_ = {}
    words_by_user_ = {}

    for id in users:
        if id in words_by_user:
            words_by_user_[user_dict[id]] = words_by_user[id]
    for id in items:
        if id in words_by_item:
            words_by_item_[item_dict[id]] = words_by_item[id]

    del words_by_item, words_by_user

    # データフレームのidを数値にする
    #train
    train_df["reviewerID"] = [user_dict[id] for id in train_df["reviewerID"].tolist()]
    train_df["asin"] = [item_dict[id] for id in train_df["asin"].tolist()]
    #valid
    valid_df["reviewerID"] = [user_dict[id] for id in valid_df["reviewerID"].tolist()]
    valid_df["asin"] = [item_dict[id] for id in valid_df["asin"].tolist()]
    #test
    test_df["reviewerID"] = [user_dict[id] for id in test_df["reviewerID"].tolist()]
    test_df["asin"] = [item_dict[id] for id in test_df["asin"].tolist()]

    del notParsedWords

    pd.DataFrame(words_by_user_.items(), columns=["user_id", "words"]).to_csv(
        "data/words_by_user/" + file, index=False)
    pd.DataFrame(words_by_item_.items(), columns=["item_id", "words"]).to_csv(
        "data/words_by_item/" + file, index=False)
    train_df.to_csv("data/train/" + file, index=False)
    valid_df.to_csv("data/valid/" + file, index=False)
    test_df.to_csv("data/test/" + file, index=False)
