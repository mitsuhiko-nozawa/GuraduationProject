import json
import os
import glob
import gzip
from io import StringIO

import numpy as np
import pandas as pd


def data_loder1(d_path, fname, DROP_NUM = 1):
    """
    AMAZONとyelpのデータの前処理
    ・jsonの解凍、処理
    ・カラム名を揃える
    ・データ抽出する
    """

    print("== start ========", fname, "file is being processed ===============")
    # gzipから解凍
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
    fname_save = d_path.replace("raw", "processed2") + fname.replace(".json.gz", "__") + str(DROP_NUM) + ".csv.gz"
    df.to_csv(fname_save, index=False)

    print("== end =========", fname, " is completely processed ===============")
    print("")




def data_loder2(d_path, fname, DROP_NUM = 1):
    print("== start ========", fname, "file is being processed ===============")
    df = pd.read_csv(d_path+fname)
    print(" df shape is ", df.shape)

    # 必要なカラムだけ取り出す
    drop_col = []
    rename_dic = {}
    if "bgg" in fname:
        drop_col = ['Unnamed: 0', 'ID']
        rename_dic = {
            "name": "asin",
            "user": "reviewerID",
            "comment": "reviewText",
            "rating": "overall"
        }

    if "hotel" in fname:
        drop_col = ['Date', 'NoOfReaders',
                    'HelpfulToNoOfreaders', 'Value_rating', 'Rooms_rating',
                    'Location_rating', 'Cleanliness_rating', 'Checkin_rating',
                    'Service_rating', 'Businessservice_rating',
                    'AveragePricing']
        rename_dic = {
            "Hotelid": "asin",
            "userid": "reviewerID",
            "reviewtext": "reviewText",
            "AverageOverallRatingOfHotel" : "overall"
        }

    df.drop(columns=drop_col, inplace=True)
    df.rename(columns=rename_dic, inplace=True)
    print(df.columns)


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
    fname_save = d_path.replace("raw", "processed2") + fname.replace(".csv.gz", "__") + str(DROP_NUM) + ".csv.gz"
    df.to_csv(fname_save, index=False)

    print("== end ==========", fname, " is completely processed ===============")
    print("")