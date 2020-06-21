import json
import os
import glob
import gzip
from io import StringIO

import numpy as np
import pandas as pd



def data_loder(fname):
    """
    data description
    amazon
    https://nijianmo.github.io/amazon/index.html

    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    vote - helpful votes of the review
    style - a disctionary of the product metadata, e.g., "Format" is "Hardcover"
    reviewText - text of the review
    overall - rating of the product
    summary - summary of the review
    unixReviewTime - time of the review (unix time)
    reviewTime - time of the review (raw)
    image - images that users post after they have received the product

    yelp
    https://www.yelp.com/dataset/documentation/main
    review_id
    user_id
    business_id
    stars
    date
    text
    useful
    funny
    cool
    """

    """
    関数の仕様
    ・jsonの解凍、処理
    ・カラム名を揃える
    ・5件以上評価のあるユーザ、アイテムに抽出する
    
    
    
    """

    print("===============", fname, "file is being processed ===============")

    # gzipから解凍
    d_path = "./Dataset/raw/"
    lines = []
    with gzip.open(d_path + fname, "rt") as gzip_open:
        while True:
            line = gzip_open.readline()
            if len(line) <= 1:
                # print("numner of data is ", len(line))
                break
            lines.append(line)

    print(" successfully unzip file")

    #　解凍後、必要な絡むだけ抽出する
    li = []
    for line in lines:
        #try:
        io = StringIO(line)
        json_load = json.load(io)
        # pandas に入れる前にここで消しちゃおう
        if "AMAZON" in fname:
            drop_cols = [
                'verified', 'reviewTime', 'style',
                'reviewerName', 'summary', 'unixReviewTime', 'vote',
                'image'
            ]
            for col in drop_cols:
                if col in json_load:
                    del json_load[col]

        else: #yelp
            json_load["reviewerID"] = json_load["user_id"]
            json_load["asin"] = json_load["business_id"]
            json_load["overall"] = json_load["stars"]
            json_load["reviewText"] = json_load["text"]
            drop_cols = [
                'review_id', 'date', 'useful',
                'funny', 'cool', 'user_id', 'business_id',
                'stars', 'text'
            ]
            for col in drop_cols:
                if col in json_load:
                    del json_load[col]
        li.append(json_load)
        #except:
        #    pass
    print(" successfully drop unnecessary columns")
    del lines



    df = pd.DataFrame(li)
    print(" df shape is ", df.shape)
    # df.to_csv(fname.replace(".json.gz", ".csv"), index=False)
    del li

    df_ = df.copy()  # あとでテキストを参照する


    # asinが同じだが種類が異なるアイテムを落とす、テキストはここで落ちてしまう
    df = pd.DataFrame(df.groupby(["reviewerID", "asin"], as_index=False).mean())


    # 評価数の少ないユーザ、アイテムを落とす
    DROP_NUM = 1
    drop_users = df.groupby(["reviewerID"]).filter(lambda group: group["overall"].count() < DROP_NUM)["reviewerID"].unique()
    drop_items = df.groupby(["asin"]).filter(lambda group: group["overall"].count() < DROP_NUM)["asin"].unique()
    print(" drop user : ", len(drop_users))
    print(" drop item : ", len(drop_items))

    for userid in drop_users:
        df = df[df["reviewerID"] != userid]
    for itemid in drop_items:
        df = df[df["reviewerID"] != itemid]
    print(" df shape is ", df.shape)


    # テキストをくっつける
    for idx, row in df.iterrows():
        userid = row["reviewerID"]
        itemid = row["asin"]
        tmp = df_[df_["reviewerID"] == userid][df_["asin"] == itemid].iloc[0]
        text = tmp["reviewText"]
        overall = tmp["overall"]
        df.at[idx, "reviewText"] = text
        df.at[idx, "overall"] = overall

    print(" final df shape is ", df.shape)


    """
    
    #drop user if user_num < 5
    drop_user = []
    user_dict = {}
    uni_user = df["reviewerID"].unique()
    for id in uni_user:
        user_dict[id] = []
    for i, row in df.iterrows():
        id = row["reviewerID"]
        user_dict[id].append(i)
    for key, val in user_dict.items():
        if len(val) < 5:
            for idx in val:
                drop_user.append(idx)

    df.drop(index=drop_user, inplace=True)
    print("user")
    del drop_user, user_dict, uni_user

    #drop item if item_num < 5
    drop_item = []
    item_dict = {}
    uni_item = df["asin"].unique()

    for id in uni_item:
        print("id")
        item_dict[id] = []
    for i, row in df.iterrows():
        print("row")
        id = row["asin"]
        item_dict[id].append(i)
    for key, val in item_dict.items():
        print("key")
        if len(val) < 5:
            for idx in val:
                drop_item.append(idx)
    df.drop(index=drop_item, inplace=True)
    print("item")
    del drop_item, item_dict, uni_item


    user_list = df["reviewerID"].unique()
    for id in user_list:
        print(id)
        temp = df[df["reviewerID"] == id]
        item_list = temp["asin"].unique()
        ids = temp.index
        for idx, asin in zip(ids, item_list):
            temp2 = df_[(df_["reviewerID"] == id) & (df_["asin"] == asin)]
            overall = temp2["overall"].values[0]
            text = temp2["reviewText"].tolist()[0]
            df.at[idx, "overall"] = overall
            df.at[idx, "reviewText"] = text
    del user_list
    """



    fname_save = d_path.replace("raw", "processed") + fname.replace(".json.gz", ".csv")
    df.to_csv(fname_save, index=False)
    print("===============", fname, " is completely processed ===============")
    print("")


if __name__ == "__main__":
    data_list = os.listdir("./Dataset/raw/")
    print(os.listdir("./Dataset/processed/"))
    print(data_list)
    for fname in data_list:
        if ".gz" in fname:

            if "AMAZON" in fname:
                data_loder(fname)
            elif "yelp" in fname:
                continue
                #data_loder(fname)



