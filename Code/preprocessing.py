import json
import os
import glob
import gzip
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib


def data_loder(fname):
    #fname_read = "AMAZON_FASHION_5.json.gz"

    """
    data description

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

    cite from
    amazon
    https://nijianmo.github.io/amazon/index.html

    yelp
    https://www.yelp.com/dataset/documentation/main

    jester
    http://eigentaste.berkeley.edu/dataset/

    """

    d_path = "./Dataset/raw/"
    with gzip.open(d_path+fname, "rt") as gzip_open:
        lines = gzip_open.read().split('\n')

        li = []
        for line in lines:
            try:
                io = StringIO(line)
                json_load = json.load(io)
                li.append(json_load)
            except:
                pass


    df = pd.DataFrame(li)

    fname_save = d_path.replace("raw", "processed") + fname.replace(".json.gz", ".csv")
    df.to_csv(fname_save, index=False)
    print(fname_save, " is complete ")

    return df


if __name__ == "__main__":
    data_list = os.listdir("./Dataset/raw/")
    print(os.listdir("./Dataset/processed/"))
    print(data_list)
    for fname in data_list:
        if "AMAZON" in fname:
            data_loder(fname)



