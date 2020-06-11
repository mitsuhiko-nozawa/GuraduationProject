#このファイルでしか使わないものをここに書けばいいよ

if __name__ == "__main__":
    import os
    import random
    import numpy as np
    import pandas as pd
    from models.basic_CF import cf_trainer
    from models.MF import mf_trainer
    np.random.seed(0)
    random.seed(0)

    print(os.listdir("../"))
    df = pd.read_csv("../Dataset/processed/AMAZON_FASHION_5.csv")


    print("data load is complete")
    drop_cols = [
        'verified', 'reviewTime', 'style',
        'reviewerName', 'summary', 'unixReviewTime', 'vote',
        'image'
    ]
    for col in drop_cols:
        df.drop(col, axis=1, inplace=True)

    df = pd.DataFrame(df.groupby(["reviewerID", "asin"], as_index=False).mean())
    #self.df.fillna(0, inplace=True)

    # データフレームからmatrixに変形しないと、traintestsplitできない
    # splitはランダムにレーティングを切り抜いて行う
    # validとtestのデータの形に迷う
    # とりあえず行列で

    # train test split
    train_df = df.copy()
    valid_df = df.copy()
    test_df = df.copy()
    valid_df["overall"] = 0.0
    test_df["overall"] = 0.0

    gross_num = df.shape[0]
    valid_num = int(np.ceil(gross_num*0.1))
    test_num = int(np.ceil(gross_num*0.1))
    train_num = gross_num - valid_num - test_num

    train_idx_c = random.sample(range(0, gross_num), valid_num+test_num)
    valid_idx = random.sample(train_idx_c, valid_num)
    test_idx = list(set(train_idx_c) - set(valid_idx))

    valid_df.iloc[valid_idx, df.columns.get_loc("overall")] = df.iloc[valid_idx, df.columns.get_loc("overall")].values
    test_df.iloc[test_idx, df.columns.get_loc("overall")] = df.iloc[test_idx, df.columns.get_loc("overall")].values
    train_df.iloc[train_idx_c, df.columns.get_loc("overall")] = 0.0

    #データを分割した時に、全ての行またはカラムが0になった場合、0除算が起きてしまう



    cf = cf_trainer()
    cf.train(train_df, valid_df)
    cf.predict(test_df)
    cf.evaluate()
    # test: rmse is  2.743305121444837 , mae is  2.1584022138638783


    mf = mf_trainer()
    mf.train(train_df, valid_df, dim_size=256, r=0.1, lr=0.000008, epoch_size=500, step=10, early_stopping=True)
    mf.predict(test_df)
    mf.evaluate()
    # lr = 0.8, dimsize = 200
    # test  : rmse is  2.1738787782887385 , mae is  1.4337459796348844