if __name__ == "__main__":
    import os


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


