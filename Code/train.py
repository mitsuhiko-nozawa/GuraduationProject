#このファイルでしか使わないものをここに書けばいいよ
import re
def parse(parser, df, embedding_dim):
    text_by_user = {}
    text_by_item = {}
    notParsedWords = []
    for userid, itemid, text in zip(df["reviewerID"].tolist(), df["asin"].tolist(), df["reviewText"].tolist()):
        if userid not in text_by_user:
            text_by_user[userid] = []
        if itemid not in text_by_item:
            text_by_item[itemid] = []

        if text != text:
            continue

        words = text.split()
        parsed_text = []
        for word in words:
            extracteds = re.findall(r'[a-zA-Z\']+', word)
            for extracted in extracteds:
                try:
                    parsed_text.append(parser[extracted][:embedding_dim])
                except:
                    try:
                        parsed_text.append(parser[extracted.lower()][:embedding_dim])
                    except:
                        notParsedWords.append(extracted)

        text_by_user[userid] += parsed_text
        text_by_item[itemid] += parsed_text
    for key, val in text_by_user.items():
        text_by_user[key] = np.array(val)
    for key, val in text_by_item.items():
        text_by_item[key] = np.array(val)
    return text_by_user, text_by_item, notParsedWords


if __name__ == "__main__":
    import os
    import random
    import numpy as np
    import pandas as pd
    from gensim.models import KeyedVectors
    from models.basic_CF import cf_trainer
    from models.MF import mf_trainer
    from models.DeepCoNN import DeepCoNN_trainer
    np.random.seed(0)
    random.seed(0)


    # データはすでに加工済み
    # 各ユーザ、アイテムが1回以上評価されているものを抽出
    df = pd.read_csv("../Dataset/processed/AMAZON_FASHION.csv")
    print("data is successfuly loaded")


    #ユーザとアイテムのidを取得
    user_dict = {}
    item_dict = {}
    data_size = df.shape[0]
    for i, id in enumerate(df["reviewerID"].unique()):
        user_dict[id] = i
    for i, id in enumerate(df["asin"].unique()):
        item_dict[id] = i


    # train, valid, testにスプリット
    # validとtestのインデックスを抽出
    valid_size = test_size = int(data_size*0.15)
    val_tes_index = random.sample(range(data_size), valid_size+test_size)
    val_index = random.sample(val_tes_index, valid_size)
    test_index = list(set(val_tes_index) - set(val_index))

    valid_df = df.iloc[val_index]
    test_df = df.iloc[test_index]
    train_df = df.drop(index=val_tes_index)
    del val_tes_index, df, val_index, test_index, valid_size





    # 1件もテキストを持たない人を削除
    """
    drop_user = [], drop_item = []
    df2 = train_df.groupby("reviewerID").count()
    drop_user += df2[df2["reviewText"] == 0].index.tolist()

    df2 = train_df.groupby("asin").count()
    drop_item += df2[df2["reviewText"] == 0].index.tolist()
    del df2
    """


    # ここでparseしないとあとで大変
    # {userid1 : doc, userid2, doc...}
    # {itemid1 : doc, itemid2, doc...}
    # の形のtextのテキストデータをtrain, text, testぶん作る

    notParsedWords = []
    path = "../Dataset/Corpus/GoogleNews-vectors-negative300.bin.gz"
    parser = KeyedVectors.load_word2vec_format(path, binary=True)
    embedding_dim = 50

    train_text_by_user, train_text_by_item, trainNotParsedWords = parse(parser, train_df, embedding_dim)
    valid_text_by_user, valid_text_by_item, validNotParsedWords = parse(parser, valid_df, embedding_dim)
    test_text_by_user, test_text_by_item, testNotParsedWords = parse(parser, test_df, embedding_dim)
    notParsedWords += trainNotParsedWords
    notParsedWords += validNotParsedWords
    notParsedWords += testNotParsedWords
    notParsedWords = list(set(notParsedWords))
    del parser, trainNotParsedWords, validNotParsedWords, testNotParsedWords

    # dropするよ
    drop_user = []
    drop_item = []

    for id, val in train_text_by_user.items():
        if(len(val) == 0):
            drop_user.append(id)

    for id, val in train_text_by_item.items():
        if(len(val) == 0):
            drop_item.append(id)

    for id, val in valid_text_by_user.items():
        if(len(val) == 0):
            drop_user.append(id)

    for id, val in valid_text_by_item.items():
        if(len(val) == 0):
            drop_item.append(id)

    for id, val in test_text_by_user.items():
        if(len(val) == 0):
            drop_user.append(id)

    for id, val in test_text_by_item.items():
        if(len(val) == 0):
            drop_item.append(id)


    drop_user = list(set(drop_user))
    drop_item = list(set(drop_item))


    print("user num not in train is ", len(drop_user))
    print("item num not in train is ", len(drop_item))


    # valid, testから削除
    train_df.drop(index=train_df[train_df["reviewerID"].isin(drop_user)].index, inplace=True)
    train_df.drop(index=train_df[train_df["asin"].isin(drop_item)].index, inplace=True)
    valid_df.drop(index=valid_df[valid_df["reviewerID"].isin(drop_user)].index, inplace=True)
    valid_df.drop(index=valid_df[valid_df["asin"].isin(drop_item)].index, inplace=True)
    test_df.drop(index=test_df[test_df["reviewerID"].isin(drop_user)].index, inplace=True)
    test_df.drop(index=test_df[test_df["asin"].isin(drop_item)].index, inplace=True)

    del drop_user, drop_item



    # 分割した際にtrainに一件もデータがない人、アイテムがないか確認する
    drop_user = list(set(user_dict.keys()) - set(train_df["reviewerID"].unique()))
    drop_item = list(set(item_dict.keys()) - set(train_df["asin"].unique()))

    print("user num not in train is ", len(drop_user))
    print("item num not in train is ", len(drop_item))

    # valid, testから削除
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
    for i, id in enumerate(train_df["reviewerID"].unique()):
        user_dict[id] = i
    for i, id in enumerate(train_df["asin"].unique()):
        item_dict[id] = i

    # text dictのkeyを振り直し, maxlen
    maxlen = 0
    train_text_by_user_ = {}
    train_text_by_item_ = {}
    valid_text_by_user_ = {}
    valid_text_by_item_ = {}
    test_text_by_user_ = {}
    test_text_by_item_ = {}

    for id, val in train_text_by_user.items():
        if id in user_dict:
            train_text_by_user_[user_dict[id]] = val
            maxlen = max(maxlen, val.shape[0])

    for id, val in train_text_by_item.items():
        if id in item_dict:
            train_text_by_item_[item_dict[id]] = val
            maxlen = max(maxlen, val.shape[0])

    for id, val in valid_text_by_user.items():
        if id in user_dict:
            valid_text_by_user_[user_dict[id]] = val
            maxlen = max(maxlen, val.shape[0])

    for id, val in valid_text_by_item.items():
        if id in item_dict:
            valid_text_by_item_[item_dict[id]] = val
            maxlen = max(maxlen, val.shape[0])

    for id, val in test_text_by_user.items():
        if id in user_dict:
            test_text_by_user_[user_dict[id]] = val
            maxlen = max(maxlen, val.shape[0])

    for id, val in test_text_by_item.items():
        if id in item_dict:
            test_text_by_item_[item_dict[id]] = val
            maxlen = max(maxlen, val.shape[0])



    del train_text_by_user, train_text_by_item, valid_text_by_user, valid_text_by_item, \
        test_text_by_user, test_text_by_item

    # dfで渡すと重いのでリストに変換
    # train = { col1 : [], col2 : [], ...., coln : [] }
    train = {}
    valid = {}
    test = {}
    #train
    train["reviewerID"] = [user_dict[id] for id in train_df["reviewerID"].tolist()]
    train["asin"] = [item_dict[id] for id in train_df["asin"].tolist()]
    train["overall"] = train_df["overall"].values
    train["reviewText"] = train_df["reviewText"].tolist()

    #valid
    valid["reviewerID"] = [user_dict[id] for id in valid_df["reviewerID"].tolist()]
    valid["asin"] = [item_dict[id] for id in valid_df["asin"].tolist()]
    valid["overall"] = valid_df["overall"].values
    valid["reviewText"] = valid_df["reviewText"].tolist()

    #test
    test["reviewerID"] = [user_dict[id] for id in test_df["reviewerID"].tolist()]
    test["asin"] = [item_dict[id] for id in test_df["asin"].tolist()]
    test["overall"] = test_df["overall"].values
    test["reviewText"] = test_df["reviewText"].tolist()

    del train_df, valid_df, test_df


    train_size = len(train["asin"])


    # 学習部分
    # print("")
    # print("cf training")
    # cf = cf_trainer(user_dict, item_dict)
    # cf.train(train)
    # cf.predict(test)
    # cf.evaluate()

    # AMAZON
    # test: rmse is  3.2003530401919904 , mae is  2.773360972887591
    # del cf

    #print("")
    #print("mf training")
    #mf = mf_trainer(user_dict, item_dict)
    #mf.train(train, valid, dim_size=20, r=0.1, lr=0.0008, epoch_size=10000, desc_step=50, early_stopping=True)
    #mf.predict(test)
    #mf.evaluate()
    # lr=0.0001, dimsize = 200
    # test  : rmse is  2.1738787782887385 , mae is  1.4337459796348844
    """
    dim_size が少ない方がいい結果になる
    500epoch
    1で1.6, 2で1.7, 10で2.0, 20で
    
    
    """

    #del mf

    print("")
    print("DeepCoNN training")
    DeepCoNN = DeepCoNN_trainer(user_dict, item_dict,
                                train_text_by_user_, train_text_by_item_,
                                valid_text_by_user_, valid_text_by_item_,
                                test_text_by_user_, test_text_by_item_,
                                embedding_dim=embedding_dim, filter_num=100, filter_size=3, feature_size=50, maxlen = maxlen,
                                batch_size=32, epoch=1)
    DeepCoNN.train(train, valid)
    DeepCoNN.predict(test)
    DeepCoNN.evaluate()
    del DeepCoNN


