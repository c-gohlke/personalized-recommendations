import os
import pandas as pd
import numpy as np
import torch
import pickle
from scipy import sparse

from params import (
    DS,
    OG_DATA_PATH,
    PROCESSED_DATA_PATH,
    PROCESSED_DATA_OUT_PATH,
    OUT_PATH,
    BASE_PATH,
)


class DataProcessor:
    def __init__(self):
        self.load_customer_df(
            OG_DATA_PATH, PROCESSED_DATA_PATH, PROCESSED_DATA_OUT_PATH
        )
        self.load_article_df(OG_DATA_PATH, PROCESSED_DATA_PATH, PROCESSED_DATA_OUT_PATH)
        self.load_transaction_df(
            OG_DATA_PATH, PROCESSED_DATA_PATH, PROCESSED_DATA_OUT_PATH
        )
        self.get_customer_article_count()
        self.get_train_df()
        self.get_test_df()
        self.get_test_gts()
        self.create_splits()

    def load_customer_df(
        self, og_data_path, processed_data_path, processed_data_out_path
    ):
        if (
            os.path.exists(os.path.join(processed_data_path, "customer_df.pickle"))
            and os.path.exists(
                os.path.join(processed_data_path, "old_keys_to_new_customers.pickle")
            )
            and os.path.exists(
                os.path.join(processed_data_path, "new_keys_to_old_customers.pickle")
            )
        ):
            print("customer data fast load")
            with open(
                os.path.join(processed_data_path, "customer_df.pickle"), "rb"
            ) as handle:
                self.customer_df = pickle.load(handle)
            with open(
                os.path.join(processed_data_path, "old_keys_to_new_customers.pickle"),
                "rb",
            ) as handle:
                self.old_keys_to_new_customers = pickle.load(handle)
            with open(
                os.path.join(processed_data_path, "new_keys_to_old_customers.pickle"),
                "rb",
            ) as handle:
                self.new_keys_to_old_customers = pickle.load(handle)
        else:
            print("generating customer data")
            if not os.path.exists(processed_data_out_path):
                os.makedirs(processed_data_out_path)
            with open(os.path.join(og_data_path, "customers.csv")) as f:
                customer_df = pd.read_csv(f)
                if DS == "only_test" or DS == "only_test_customers":
                    if not os.path.exists(
                        os.path.join(OUT_PATH, "test_df_info.pickle")
                    ):
                        self.find_test_info()
                    with open(
                        os.path.join(OUT_PATH, "test_df_info.pickle"), "rb"
                    ) as handle:
                        test_customer_ids = pickle.load(handle)["customer_ids"]

                    customer_df = customer_df[
                        customer_df["customer_id"].isin(test_customer_ids)
                    ]

                self.customer_df = customer_df.sort_values(by="customer_id")
                self.old_keys_to_new_customers = {
                    k: i for (i, k) in enumerate(self.customer_df["customer_id"].values)
                }  # TODO check uniqueness
                self.new_keys_to_old_customers = {
                    i: c for (i, c) in enumerate(self.customer_df["customer_id"].values)
                }

                new_customer_ids = range(len(self.customer_df))
                self.customer_df["customer_id"] = new_customer_ids

                if DS == "small":
                    self.customer_df = self.customer_df[
                        self.customer_df["customer_id"] < 1000
                    ]
                with open(
                    os.path.join(processed_data_out_path, "customer_df.pickle"), "wb"
                ) as handle:
                    pickle.dump(
                        self.customer_df, handle, protocol=pickle.HIGHEST_PROTOCOL
                    )

                with open(
                    os.path.join(
                        processed_data_out_path, "old_keys_to_new_customers.pickle"
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
                        processed_data_out_path, "new_keys_to_old_customers.pickle"
                    ),
                    "wb",
                ) as handle:
                    pickle.dump(
                        self.new_keys_to_old_customers,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

    def load_article_df(
        self, og_data_path, processed_data_path, processed_data_out_path
    ):
        if (
            os.path.exists(os.path.join(processed_data_path, "article_df.pickle"))
            and os.path.exists(
                os.path.join(processed_data_path, "old_keys_to_new_articles.pickle")
            )
            and os.path.exists(
                os.path.join(processed_data_path, "new_keys_to_old_articles.pickle")
            )
        ):
            print("article data fast load")
            with open(
                os.path.join(processed_data_path, "article_df.pickle"), "rb"
            ) as handle:
                self.article_df = pickle.load(handle)
            with open(
                os.path.join(processed_data_path, "old_keys_to_new_articles.pickle"),
                "rb",
            ) as handle:
                self.old_keys_to_new_articles = pickle.load(handle)
            with open(
                os.path.join(processed_data_path, "new_keys_to_old_articles.pickle"),
                "rb",
            ) as handle:
                self.new_keys_to_old_articles = pickle.load(handle)
        else:
            print("generating article data")
            with open(os.path.join(og_data_path, "articles.csv")) as f:
                article_df = pd.read_csv(f)
                if DS == "only_test":
                    if not os.path.exists(
                        os.path.join(OUT_PATH, "test_df_info.pickle")
                    ):
                        self.find_test_info()
                    with open(
                        os.path.join(OUT_PATH, "test_df_info.pickle"), "rb"
                    ) as handle:
                        test_article_ids = pickle.load(handle)["article_ids"]

                    article_df = article_df[
                        article_df["article_id"].isin(test_article_ids)
                    ]

            self.article_df = article_df.sort_values(by="article_id")
            self.old_keys_to_new_articles = {
                c: i for (i, c) in enumerate(self.article_df["article_id"].values)
            }
            self.new_keys_to_old_articles = {
                i: c for (i, c) in enumerate(self.article_df["article_id"].values)
            }
            new_article_ids = range(len(self.article_df))
            self.article_df["article_id"] = new_article_ids

            if DS == "small":
                self.article_df = self.article_df[self.article_df["article_id"] < 1000]

            with open(
                os.path.join(processed_data_out_path, "article_df.pickle"), "wb"
            ) as handle:
                pickle.dump(self.article_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(
                os.path.join(
                    processed_data_out_path, "old_keys_to_new_articles.pickle"
                ),
                "wb",
            ) as handle:
                pickle.dump(
                    self.old_keys_to_new_articles,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            with open(
                os.path.join(
                    processed_data_out_path, "new_keys_to_old_articles.pickle"
                ),
                "wb",
            ) as handle:
                pickle.dump(
                    self.new_keys_to_old_articles,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

    def load_transaction_df(
        self, og_data_path, processed_data_path, processed_data_out_path
    ):
        if os.path.exists(os.path.join(processed_data_path, "transaction_df.pickle")):
            print("transaction data fast load")
            with open(
                os.path.join(processed_data_path, "transaction_df.pickle"), "rb"
            ) as handle:
                self.transaction_df = pickle.load(handle)
        else:
            print("generating transaction data")
            with open(os.path.join(og_data_path, "transactions_train.csv")) as f:
                self.transaction_df = pd.read_csv(f)
                if DS == "only_test" or DS == "only_test_customers":
                    if not os.path.exists(
                        os.path.join(OUT_PATH, "test_df_info.pickle")
                    ):
                        self.find_test_info()
                    with open(
                        os.path.join(OUT_PATH, "test_df_info.pickle"), "rb"
                    ) as handle:
                        test_df_info = pickle.load(handle)
                    test_df_customers = pd.Series(test_df_info["customer_ids"])
                    self.transaction_df = self.transaction_df[
                        self.transaction_df["customer_id"].isin(test_df_customers)
                    ]
                    if DS == "only_test":
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

            if DS == "small":
                self.transaction_df = self.transaction_df[
                    self.transaction_df["article_id"] < 1000
                ]
                self.transaction_df = self.transaction_df[
                    self.transaction_df["customer_id"] < 1000
                ]

            with open(
                os.path.join(processed_data_out_path, "transaction_df.pickle"), "wb"
            ) as handle:
                pickle.dump(
                    self.transaction_df, handle, protocol=pickle.HIGHEST_PROTOCOL
                )

        self.last_day = max(self.transaction_df["t_dat"])
        self.train_threshold = self.last_day - pd.Timedelta(days=7)

    def find_test_info(self):
        processed_data_path = os.path.join(BASE_PATH, "out", "full", "data")
        processed_data_out_path = os.path.join(BASE_PATH, "out", "full", "data")

        self.load_customer_df(
            OG_DATA_PATH, processed_data_path, processed_data_out_path
        )
        self.load_article_df(OG_DATA_PATH, processed_data_path, processed_data_out_path)
        self.load_transaction_df(
            OG_DATA_PATH, processed_data_path, processed_data_out_path
        )
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

        with open(os.path.join(OUT_PATH, "test_df_info.pickle"), "wb") as handle:
            pickle.dump(test_df_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if not (
            processed_data_path == PROCESSED_DATA_PATH
            and processed_data_out_path == PROCESSED_DATA_OUT_PATH
        ):
            # reload necessary dataset
            self.__init__()

    def get_train_df(self):
        self.train_df = self.transaction_df[
            self.transaction_df["t_dat"] < self.train_threshold
        ]

        self.train_customer_ids = self.train_df["customer_id"].unique()

    def get_test_df(self):
        self.test_df = self.transaction_df[
            self.transaction_df["t_dat"] >= self.train_threshold
        ]
        self.test_customer_ids = self.test_df["customer_id"].unique()

    def get_article_df(self):
        return self.article_df

    def get_customer_df(self):
        return self.customer_df

    def get_customer_article_count(self):
        print("calculating customer and article counts")
        self.customer_ids = self.customer_df["customer_id"].unique()
        self.article_ids = self.article_df["article_id"].unique()
        self.customer_count = len(self.customer_ids)
        self.article_count = len(self.article_ids)

    def create_splits(self):
        if os.path.exists(os.path.join(PROCESSED_DATA_PATH, "S_train.pickle")):
            print("S_train fast load")
            with open(
                os.path.join(PROCESSED_DATA_PATH, "S_train.pickle"), "rb"
            ) as handle:
                self.S_train = pickle.load(handle)
        else:
            print("generate S_train")
            self.S_train = sparse.csr_matrix(
                (self.customer_count, self.article_count), dtype=np.int8
            )
            train_cids = self.train_df["customer_id"].values
            train_aids = self.train_df["article_id"].values

            self.S_train[train_cids, train_aids] = 1

            with open(
                os.path.join(PROCESSED_DATA_OUT_PATH, "S_train.pickle"), "wb",
            ) as handle:
                pickle.dump(self.S_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if os.path.exists(os.path.join(PROCESSED_DATA_PATH, "S_test.pickle")):
            print("S_test fast load")
            with open(
                os.path.join(PROCESSED_DATA_PATH, "S_test.pickle"), "rb"
            ) as handle:
                self.S_test = pickle.load(handle)
        else:
            print("generate S_test")
            self.S_test = sparse.csr_matrix(
                (self.customer_count, self.article_count), dtype=np.int8
            )
            test_cids = self.test_df["customer_id"].values
            test_aids = self.test_df["article_id"].values
            self.S_test[test_cids, test_aids] = 1

            with open(
                os.path.join(PROCESSED_DATA_OUT_PATH, "S_test.pickle"), "wb"
            ) as handle:
                pickle.dump(self.S_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_test_gts(self):
        if os.path.exists(os.path.join(PROCESSED_DATA_PATH, "test_gts.pickle")):
            print("get_test_gt fast load")
            with open(
                os.path.join(PROCESSED_DATA_PATH, "test_gts.pickle"), "rb"
            ) as handle:
                self.test_gts = pickle.load(handle)
        else:
            self.test_gts = []
            for customer_id in self.test_customer_ids:
                gt = self.test_df[self.test_df["customer_id"] == customer_id][
                    "article_id"
                ].values
                self.test_gts.append(gt)

            with open(
                os.path.join(PROCESSED_DATA_OUT_PATH, "test_gts.pickle"), "wb",
            ) as handle:
                pickle.dump(self.test_gts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_new_loader(self, batch_size):
        train_data = self.sample_train()
        train_data = torch.Tensor(train_data).int()
        train_set = Dataset(train_data)
        return torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, pin_memory=True,
        )

    def sample_train(self):
        X = np.empty((len(self.train_customer_ids), 3))

        X[:, 0] = self.train_customer_ids
        for i in range(len(self.train_customer_ids)):
            cid = self.train_customer_ids[i]
            rated_items = self.S_train[cid, :].nonzero()[1]

            X[i, 1] = np.random.choice(rated_items, 1).item()
            while True:
                rand_item = np.random.randint(0, self.S_train.shape[1])
                if rand_item not in rated_items:
                    X[i, 2] = rand_item
                    break
        return X


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


#%%
if __name__ == "__main__":
    # from params import OG_DATA_PATH, PROCESSED_DATA_PATH, PROCESSED_DATA_OUT_PATH

    dataprocessor = DataProcessor()
    #%%
