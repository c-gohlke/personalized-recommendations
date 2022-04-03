import os
import pandas as pd
import numpy as np
import pickle
from scipy import sparse
from params import BASE_PATH, OG_DATA_NAME

OG_PATH = os.path.join(BASE_PATH, OG_DATA_NAME)
OUT_PATH = os.path.join(BASE_PATH, "out")


class DataProcessor:
    def __init__(self):
        print("loading og data")
        self.load_og_data()

        ds = "full"
        test_customer_ids, test_article_ids = self.process_ds(
            self.article_df, self.customer_df, self.transaction_df, ds
        )

        test_customer_ids = pd.Series(test_customer_ids)
        test_article_ids = pd.Series(test_article_ids)

        ds = "only_test_customers"
        customer_df = self.customer_df[
            self.customer_df["customer_id"].isin(test_customer_ids)
        ]
        transaction_df = self.transaction_df[
            self.transaction_df["customer_id"].isin(test_customer_ids)
        ]
        self.process_ds(self.article_df, customer_df, transaction_df, ds)

        ds = "only_test"
        article_df = self.article_df[
            self.article_df["article_id"].isin(test_article_ids)
        ]
        transaction_df = transaction_df[
            transaction_df["article_id"].isin(test_article_ids)
        ]
        self.process_ds(article_df, customer_df, transaction_df, ds)

    def process_ds(self, article_df, customer_df, transaction_df, ds):
        print(f"processing data for dataset {ds}")
        out_ds_path = os.path.join(OUT_PATH, ds)
        p_out_path = os.path.join(out_ds_path, "data")
        if not os.path.exists(p_out_path):
            os.makedirs(p_out_path)

            print("replacing customer ids")
            customer_df, c_transform, c_inverse_transform = self.replace_customer_ids(
                self.customer_df, p_out_path
            )
            print("replacing article ids")
            article_df, a_transform, a_inverse_transform = self.replace_article_ids(
                self.article_df, p_out_path
            )
            print("processing transactions data")
            (
                train_df,
                test_df,
                train_customer_ids,
                train_article_ids,
                test_customer_ids,
                test_article_ids,
            ) = self.process_transaction(
                transaction_df, c_transform, a_transform, p_out_path
            )

            customer_count, article_count = self.process_meta_data(
                customer_df,
                article_df,
                train_customer_ids,
                test_customer_ids,
                p_out_path,
            )
            print("processing S train")
            self.process_S_train(train_df, customer_count, article_count, p_out_path)
            print("processing train gt")
            self.process_train_gt(train_df, train_customer_ids, p_out_path)
            print("processing test gt")
            self.process_test_gt(test_df, test_customer_ids, p_out_path)
            return test_customer_ids, test_article_ids

    def load_og_data(self):
        with open(os.path.join(OG_PATH, "articles.csv")) as f:
            self.article_df = pd.read_csv(f)
        print("og articles loaded")
        with open(os.path.join(OG_PATH, "customers.csv")) as f:
            self.customer_df = pd.read_csv(f)
        print("og customers loaded")
        with open(os.path.join(OG_PATH, "transactions_train.csv")) as f:
            self.transaction_df = pd.read_csv(f)
        print("og transactions loaded")

    def replace_customer_ids(self, customer_df, p_out_path):
        c_transform = {}
        c_inverse_transform = {}
        for (i, k) in enumerate(customer_df["customer_id"].unique()):
            c_transform[k] = i
            c_inverse_transform[i] = k

        customer_df["customer_id"] = (
            customer_df["customer_id"]
            .transform(lambda x: c_transform[x])
            .astype("int64")
        )

        with open(os.path.join(p_out_path, "customer_df.pickle"), "wb") as handle:
            pickle.dump(customer_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(p_out_path, "c_transform.pickle"), "wb",) as handle:
            pickle.dump(
                c_transform, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )

        with open(
            os.path.join(p_out_path, "c_inverse_transform.pickle"), "wb",
        ) as handle:
            pickle.dump(
                c_inverse_transform, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
        return customer_df, c_transform, c_inverse_transform

    def replace_article_ids(self, article_df, p_out_path):
        a_transform = {}
        a_inverse_transform = {}
        for (i, k) in enumerate(article_df["article_id"].unique()):
            a_transform[k] = i
            a_inverse_transform[i] = k

        article_df["article_id"] = (
            article_df["article_id"].transform(lambda x: a_transform[x]).astype("int32")
        )

        with open(os.path.join(p_out_path, "article_df.pickle"), "wb") as handle:
            pickle.dump(article_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(p_out_path, "a_transform.pickle"), "wb",) as handle:
            pickle.dump(
                a_transform, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )

        with open(
            os.path.join(p_out_path, "a_inverse_transform.pickle"), "wb",
        ) as handle:
            pickle.dump(
                a_inverse_transform, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
        return article_df, a_transform, a_inverse_transform

    def process_transaction(self, transaction_df, c_transform, a_transform, p_out_path):
        transaction_df["customer_id"] = (
            transaction_df["customer_id"]
            .transform(lambda x: c_transform[x])
            .astype("int64")
        )
        transaction_df["article_id"] = (
            transaction_df["article_id"]
            .transform(lambda x: a_transform[x])
            .astype("int32")
        )

        transaction_df["t_dat"] = pd.to_datetime(transaction_df["t_dat"])
        transaction_df["price"] = transaction_df["price"].astype("float32")
        transaction_df["sales_channel_id"] = transaction_df["sales_channel_id"].astype(
            "int8"
        )

        with open(os.path.join(p_out_path, "transaction_df.pickle"), "wb") as handle:
            pickle.dump(transaction_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

        last_day = max(transaction_df["t_dat"])
        train_threshold = last_day - pd.Timedelta(days=7)

        train_df = transaction_df[transaction_df["t_dat"] < train_threshold]
        test_df = transaction_df[transaction_df["t_dat"] >= train_threshold]

        with open(os.path.join(p_out_path, "train_df.pickle"), "wb") as handle:
            pickle.dump(train_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(p_out_path, "test_df.pickle"), "wb") as handle:
            pickle.dump(test_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

        train_customer_ids = train_df["customer_id"].unique()
        train_article_ids = train_df["article_id"].unique()
        train_group_customer = train_df.groupby("customer_id")
        train_group_customer_indexes = train_group_customer.apply(lambda x: x.index)
        train_customer_articles = train_group_customer_indexes.apply(
            lambda x: train_df.loc[x]["article_id"].values
        )
        with open(
            os.path.join(p_out_path, "train_customer_articles.pickle"), "wb"
        ) as handle:
            pickle.dump(
                train_customer_articles, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

        test_customer_ids = test_df["customer_id"].unique()
        test_article_ids = test_df["article_id"].unique()
        test_group_customer = test_df.groupby("customer_id")
        test_group_customer_indexes = test_group_customer.apply(lambda x: x.index)
        test_customer_articles = test_group_customer_indexes.apply(
            lambda x: test_df.loc[x]["article_id"].values
        )
        with open(
            os.path.join(p_out_path, "test_customer_articles.pickle"), "wb"
        ) as handle:
            pickle.dump(
                test_customer_articles, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

        return (
            train_df,
            test_df,
            train_customer_ids,
            train_article_ids,
            test_customer_ids,
            test_article_ids,
        )

    def process_meta_data(
        self, customer_df, article_df, train_customer_ids, test_customer_ids, p_out_path
    ):
        customer_ids = customer_df["customer_id"].unique()
        article_ids = article_df["article_id"].unique()
        customer_count = len(customer_ids)
        article_count = len(article_ids)

        with open(os.path.join(p_out_path, "customer_ids.pickle"), "wb",) as handle:
            pickle.dump(
                customer_ids, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
        with open(os.path.join(p_out_path, "article_ids.pickle"), "wb",) as handle:
            pickle.dump(
                article_ids, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
        with open(os.path.join(p_out_path, "customer_count.pickle"), "wb",) as handle:
            pickle.dump(
                customer_count, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
        with open(os.path.join(p_out_path, "article_count.pickle"), "wb",) as handle:
            pickle.dump(
                article_count, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
        with open(
            os.path.join(p_out_path, "train_customer_ids.pickle"), "wb",
        ) as handle:
            pickle.dump(
                train_customer_ids, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
        with open(
            os.path.join(p_out_path, "test_customer_ids.pickle"), "wb",
        ) as handle:
            pickle.dump(
                test_customer_ids, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
        return customer_count, article_count

    def process_S_train(self, train_df, customer_count, article_count, p_out):
        print("generate S_train")
        S_train = sparse.csr_matrix((customer_count, article_count), dtype=np.int8)
        train_cids = train_df["customer_id"].values
        train_aids = train_df["article_id"].values

        S_train[train_cids, train_aids] = 1

        with open(os.path.join(p_out, "S_train.pickle"), "wb",) as handle:
            pickle.dump(S_train, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    dataprocessor = DataProcessor()
