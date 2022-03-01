import os
import pandas as pd
import numpy as np
import torch
import pickle
from scipy import sparse


class DataProcessor:
    def __init__(self, params):
        self.params = params
        self.process_customer_df()
        self.process_article_df()
        self.process_transaction_df()
        self.get_customer_article_count()
        self.process_test_gt()
        self.create_splits()
        self.process_meta_data()

    def process_customer_df(self):
        print("generating customer data")
        if not os.path.exists(self.params["p_out_path"]):
            os.makedirs(self.params["p_out_path"])
        with open(os.path.join(self.params["og_path"], "customers.csv")) as f:
            customer_df = pd.read_csv(f)
            if (
                self.params["ds"] == "only_test"
                or self.params["ds"] == "only_test_customers"
            ):
                if not os.path.exists(
                    os.path.join(self.params["out_path"], "test_df_info.pickle")
                ):
                    self.find_test_info()
                with open(
                    os.path.join(self.params["out_path"], "test_df_info.pickle"), "rb"
                ) as handle:
                    test_customer_ids = pickle.load(handle)["customer_ids"]

                customer_df = customer_df[
                    customer_df["customer_id"].isin(test_customer_ids)
                ]

            self.customer_df = customer_df.sort_values(by="customer_id")
            self.old_keys_to_new_customers = {
                k: i for (i, k) in enumerate(self.customer_df["customer_id"].unique())
            }
            self.new_keys_to_old_customers = {
                i: c for (i, c) in enumerate(self.customer_df["customer_id"].unique())
            }

            new_customer_ids = range(len(self.customer_df))
            self.customer_df["customer_id"] = new_customer_ids

            if self.params["ds"] == "small":
                self.customer_df = self.customer_df[
                    self.customer_df["customer_id"] < 1000
                ]
            with open(
                os.path.join(self.params["p_out_path"], "customer_df.pickle"), "wb"
            ) as handle:
                pickle.dump(self.customer_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(
                os.path.join(
                    self.params["p_out_path"], "old_keys_to_new_customers.pickle"
                ),
                "wb",
            ) as handle:
                pickle.dump(
                    self.old_keys_to_new_customers,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            with open(
                os.path.join(
                    self.params["p_out_path"], "new_keys_to_old_customers.pickle"
                ),
                "wb",
            ) as handle:
                pickle.dump(
                    self.new_keys_to_old_customers,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

    def process_article_df(self):
        print("generating article data")
        with open(os.path.join(self.params["og_path"], "articles.csv")) as f:
            article_df = pd.read_csv(f)
            if self.params["ds"] == "only_test":
                if not os.path.exists(
                    os.path.join(self.params["out_path"], "test_df_info.pickle")
                ):
                    self.find_test_info()
                with open(
                    os.path.join(self.params["out_path"], "test_df_info.pickle"), "rb"
                ) as handle:
                    test_article_ids = pickle.load(handle)["article_ids"]

                article_df = article_df[article_df["article_id"].isin(test_article_ids)]

        self.article_df = article_df.sort_values(by="article_id")
        self.old_keys_to_new_articles = {
            c: i for (i, c) in enumerate(self.article_df["article_id"].values)
        }
        self.new_keys_to_old_articles = {
            i: c for (i, c) in enumerate(self.article_df["article_id"].values)
        }
        new_article_ids = range(len(self.article_df))
        self.article_df["article_id"] = new_article_ids

        if self.params["ds"] == "small":
            self.article_df = self.article_df[self.article_df["article_id"] < 1000]

        with open(
            os.path.join(self.params["p_out_path"], "article_df.pickle"), "wb"
        ) as handle:
            pickle.dump(self.article_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(
            os.path.join(self.params["p_out_path"], "old_keys_to_new_articles.pickle"),
            "wb",
        ) as handle:
            pickle.dump(
                self.old_keys_to_new_articles, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
        with open(
            os.path.join(self.params["p_out_path"], "new_keys_to_old_articles.pickle"),
            "wb",
        ) as handle:
            pickle.dump(
                self.new_keys_to_old_articles, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )

    def process_meta_data(self):
        self.customer_ids = self.customer_df["customer_id"].unique()
        self.article_ids = self.article_df["article_id"].unique()
        self.customer_count = len(self.customer_ids)
        self.article_count = len(self.article_ids)

        with open(
            os.path.join(self.params["p_out_path"], "customer_ids.pickle"), "wb",
        ) as handle:
            pickle.dump(
                self.customer_ids, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
        with open(
            os.path.join(self.params["p_out_path"], "article_ids.pickle"), "wb",
        ) as handle:
            pickle.dump(
                self.article_ids, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
        with open(
            os.path.join(self.params["p_out_path"], "customer_count.pickle"), "wb",
        ) as handle:
            pickle.dump(
                self.customer_count, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
        with open(
            os.path.join(self.params["p_out_path"], "article_count.pickle"), "wb",
        ) as handle:
            pickle.dump(
                self.article_count, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
        with open(
            os.path.join(self.params["p_out_path"], "train_customer_ids.pickle"), "wb",
        ) as handle:
            pickle.dump(
                self.train_customer_ids, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
        with open(
            os.path.join(self.params["p_out_path"], "test_customer_ids.pickle"), "wb",
        ) as handle:
            pickle.dump(
                self.test_customer_ids, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )

    def process_transaction_df(self):
        print("generating transaction data")
        with open(os.path.join(self.params["og_path"], "transactions_train.csv")) as f:
            self.transaction_df = pd.read_csv(f)
            if (
                self.params["ds"] == "only_test"
                or self.params["ds"] == "only_test_customers"
            ):
                if not os.path.exists(
                    os.path.join(self.params["out_path"], "test_df_info.pickle")
                ):
                    self.find_test_info()
                with open(
                    os.path.join(self.params["out_path"], "test_df_info.pickle"), "rb"
                ) as handle:
                    test_df_info = pickle.load(handle)
                test_df_customers = pd.Series(test_df_info["customer_ids"])
                self.transaction_df = self.transaction_df[
                    self.transaction_df["customer_id"].isin(test_df_customers)
                ]
                if self.params["ds"] == "only_test":
                    test_df_articles = pd.Series(test_df_info["article_ids"])
                    self.transaction_df = self.transaction_df[
                        self.transaction_df["article_id"].isin(test_df_articles)
                    ]

        self.transaction_df["customer_id"] = [
            self.old_keys_to_new_customers[v]
            for v in self.transaction_df["customer_id"].values.tolist()
        ]
        self.transaction_df["article_id"] = [
            self.old_keys_to_new_articles[v]
            for v in self.transaction_df["article_id"].values.tolist()
        ]
        self.transaction_df["t_dat"] = pd.to_datetime(self.transaction_df["t_dat"])

        if self.params["ds"] == "small":
            self.transaction_df = self.transaction_df[
                self.transaction_df["article_id"] < 1000
            ]
            self.transaction_df = self.transaction_df[
                self.transaction_df["customer_id"] < 1000
            ]

        with open(
            os.path.join(self.params["p_out_path"], "transaction_df.pickle"), "wb"
        ) as handle:
            pickle.dump(self.transaction_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.last_day = max(self.transaction_df["t_dat"])
        self.train_threshold = self.last_day - pd.Timedelta(days=7)

        self.train_df = self.transaction_df[
            self.transaction_df["t_dat"] < self.train_threshold
        ]

        self.train_customer_ids = self.train_df["customer_id"].unique()
        with open(
            os.path.join(self.params["p_out_path"], "train_df.pickle"), "wb"
        ) as handle:
            pickle.dump(self.train_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.test_df = self.transaction_df[
            self.transaction_df["t_dat"] >= self.train_threshold
        ]
        self.test_customer_ids = self.test_df["customer_id"].unique()
        with open(
            os.path.join(self.params["p_out_path"], "test_df.pickle"), "wb"
        ) as handle:
            pickle.dump(self.test_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def find_test_info(self):
        if self.params["ds"] == "full":
            self.process_transaction_df()
            self.process_article_df()
            self.process_transaction_df()
            self.get_test_df()

            test_customer_ids = self.test_df["customer_id"].unique().tolist()
            test_article_ids = self.test_df["article_id"].unique().tolist()

            test_customer_ids = [
                self.new_keys_to_old_customers[cid] for cid in test_customer_ids
            ]
            test_article_ids = [
                self.new_keys_to_old_articles[aid] for aid in test_article_ids
            ]
            test_df_info = {
                "customer_ids": test_customer_ids,
                "article_ids": test_article_ids,
            }

            with open(
                os.path.join(self.params["out_path"], "test_df_info.pickle"), "wb"
            ) as handle:
                pickle.dump(test_df_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_customer_article_count(self):
        print("calculating customer and article counts")
        self.customer_ids = self.customer_df["customer_id"].unique()
        self.article_ids = self.article_df["article_id"].unique()
        self.customer_count = len(self.customer_ids)
        self.article_count = len(self.article_ids)

    def create_splits(self):
        self.get_customer_article_count()
        print("generate S_train")
        self.S_train = sparse.csr_matrix(
            (self.customer_count, self.article_count), dtype=np.int8
        )
        train_cids = self.train_df["customer_id"].values
        train_aids = self.train_df["article_id"].values

        self.S_train[train_cids, train_aids] = 1

        with open(
            os.path.join(self.params["p_out_path"], "S_train.pickle"), "wb",
        ) as handle:
            pickle.dump(self.S_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("generate S_test")
        self.S_test = sparse.csr_matrix(
            (self.customer_count, self.article_count), dtype=np.int8
        )
        test_cids = self.test_df["customer_id"].values
        test_aids = self.test_df["article_id"].values
        self.S_test[test_cids, test_aids] = 1

        with open(
            os.path.join(self.params["p_out_path"], "S_test.pickle"), "wb"
        ) as handle:
            pickle.dump(self.S_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def process_test_gt(self):
        self.test_gt = []
        for customer_id in self.test_customer_ids:
            gt = self.test_df[self.test_df["customer_id"] == customer_id][
                "article_id"
            ].values
            self.test_gt.append(gt)

        with open(
            os.path.join(self.params["p_out_path"], "test_gts.pickle"), "wb",
        ) as handle:
            pickle.dump(self.test_gt, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    from params import POSSIBLE_DATASETS, get_params

    for ds in POSSIBLE_DATASETS:
        params = get_params(ds)
        dataprocessor = DataProcessor(params)
