# -*- coding: utf-8 -*-

import sys
import numpy as np

from utils.dataprocessor import DataProcessor
from utils.dataset import Dataset
from utils.map_score import map_score

# TODO

def get_most_popular(df):
    return df["article_id"].value_counts()[:PREDICT_AMOUNT].keys()


def evaluate_popular(most_popular, test_customer_ids, test_gts):
    pred = predict_popular(most_popular, test_customer_ids)
    score = map_score(pred, test_gts)
    print(f"score is {score}")


def predict_popular(most_popular_articles, customer_ids):
    return [most_popular_articles for _ in range(len(customer_ids))]


def find_last_purchase(customer_ids, train_df):
    for cid in customer_ids:
        c_df = train_df[train_df["customer_id"] == cid]
        c_df = c_df.sort_values(by="t_dat", ascending=True)
        print(c_df)
        sys.exit()


def predict_similar(customer_ids, last_purchase):
    pass


#%% main
if __name__ == "__main__":
    #%% load data
    from params import OG_DATA_PATH

    dataprocessor = DataProcessor(OG_DATA_PATH)

    #%% get train/test_set

    train_df = Dataset(dataprocessor.get_train_df())
    test_df = Dataset(dataprocessor.get_test_df())
    article_df = Dataset(dataprocessor.get_article_df())
    customer_df = Dataset(dataprocessor.get_customer_df())

    #%% preprocess

    train_customer_ids = train_df["customer_id"].unique()
    test_customer_ids = test_df["customer_id"].unique()
    test_gts = get_test_gt(test_customer_ids, test_df)

    #%% get most popular articles

    most_popular = get_most_popular(train_df)
    evaluate_popular(most_popular, test_customer_ids, test_gts)  # 0.003

    #%% get most similar article
    last_purchase_article_id = find_last_purchase(train_customer_ids, train_df)
