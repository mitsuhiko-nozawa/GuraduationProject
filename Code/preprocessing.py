if __name__ == "__main__":
    import os
    from dataLoder.data_loder import data_loder1, data_loder2

    DROP_NUMS = [10, 5, 1]
    print(os.listdir())
    d_path = "../Dataset/raw/"
    data_list = os.listdir(d_path)
    print(data_list)
    exceptions = [
        "yelp_academic_dataset_review.json.gz",
        "AMAZON_FASHION.json.gz",
        "bgg-13m-reviews.csv.gz"
    ]
    for DROP_NUM in DROP_NUMS:
        print("DROP_NUM : ", DROP_NUM)
        for fname in data_list:
            if ".gz" or ".zip" in fname:
                if fname in exceptions:
                    data_loder2(d_path, fname, DROP_NUM=DROP_NUM)
                else:
                    data_loder1(d_path, fname, DROP_NUM=DROP_NUM)
