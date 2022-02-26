# -*- coding: utf-8 -*-
from utils.dataprocessor import DataProcessor
import os
import pickle

from params import BASE_PATH, OG_DATA_PATH

processed_data_path = os.path.join(BASE_PATH, "out", "full", "data")
processed_data_out_path = os.path.join(BASE_PATH, "out", "full", "data")


def find_test_info():
    dataprocessor = DataProcessor(
        OG_DATA_PATH, processed_data_path, processed_data_out_path
    )

    test_df = dataprocessor.get_test_df()

    test_customer_ids = test_df["customer_id"].unique().tolist()
    test_article_ids = test_df["article_id"].unique().tolist()

    test_customer_ids = [
        dataprocessor.new_keys_to_old_customers[cid] for cid in test_customer_ids
    ]
    test_article_ids = [
        dataprocessor.new_keys_to_old_articles[aid] for aid in test_article_ids
    ]
    test_df_info = {"customer_ids": test_customer_ids, "article_ids": test_article_ids}

    with open(os.path.join(BASE_PATH, "test_df_info.pickle"), "wb") as handle:
        pickle.dump(test_df_info, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    find_test_info()
