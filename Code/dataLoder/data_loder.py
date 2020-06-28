import json
import os
import glob
import gzip
from io import StringIO

import numpy as np
import pandas as pd



def data_loder(fname, DROP_NUM = 1):
    """
    AMAZONとyelpのデータの前処理
    ・jsonの解凍、処理
    ・カラム名を揃える
    ・データ抽出する
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

    print(" successfully drop unnecessary columns")
    del lines

    df = pd.DataFrame(li)
    print(" df shape is ", df.shape)
    del li

    # reviewerID, asinが同じだが種類が異なるアイテムを落とす
    df = df[~df.duplicated(subset=["reviewerID", "asin"], keep="first")]
    print("non duplicated df shape is ", df.shape)

    # 評価数がDROP_NUM未満のユーザ、アイテムを落とす
    drop_users = df.groupby(["reviewerID"]).filter(lambda group: group["overall"].count() < DROP_NUM)["reviewerID"].unique()
    drop_items = df.groupby(["asin"]).filter(lambda group: group["overall"].count() < DROP_NUM)["asin"].unique()
    print(" drop user : ", len(drop_users))
    print(" drop item : ", len(drop_items))

    df.drop(index=df[df["reviewerID"].isin(drop_users)].index, inplace=True)
    df.drop(index=df[df["asin"].isin(drop_items)].index, inplace=True)

    print(" final df shape is ", df.shape)
    fname_save = d_path.replace("raw", "processed") + fname.replace(".json.gz", "__") + str(DROP_NUM) + ".csv.gz"
    df.to_csv(fname_save, index=False)

    print("===============", fname, " is completely processed ===============")
    print("")


if __name__ == "__main__":
    DROP_NUM = 1
    print(os.listdir())
    data_list = os.listdir("./Dataset/raw/")
    print(data_list)
    for fname in data_list:
        if ".gz" in fname:
            if "AMAZON" in fname:
                data_loder(fname, DROP_NUM=DROP_NUM)
            elif "yelp" in fname:
                data_loder(fname, DROP_NUM=DROP_NUM)


