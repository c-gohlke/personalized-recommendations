import os
import numpy as np
import torch
import pickle
from utils.dataset import Dataset


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_customer_df()
        self.load_article_df()
        self.get_customer_article_count()
        self.load_train_test()
        self.load_S_data()
        # self.load_train_df()
        # self.load_test_df()

    def load_customer_df(self):
        print("customer data fast load")
        with open(os.path.join(self.data_path, "customer_df.pickle"), "rb") as handle:
            self.customer_df = pickle.load(handle)

    def load_article_df(self):
        print("article data fast load")
        with open(os.path.join(self.data_path, "article_df.pickle"), "rb") as handle:
            self.article_df = pickle.load(handle)

    def load_train_test(self):
        # TODO improve
        import pandas as pd

        with open(
            os.path.join(self.data_path, "transaction_df.pickle"), "rb"
        ) as handle:
            self.transaction_df = pickle.load(handle)

        self.last_day = max(self.transaction_df["t_dat"])
        self.train_threshold = self.last_day - pd.Timedelta(days=7)
        self.test_df = self.transaction_df[
            self.transaction_df["t_dat"] >= self.train_threshold
        ]

        self.test_customer_ids = self.test_df["customer_id"].unique()

        self.train_df = self.transaction_df[
            self.transaction_df["t_dat"] < self.train_threshold
        ]

        self.train_customer_ids = self.train_df["customer_id"].unique()

    def get_test_df(self):
        self.test_df = self.transaction_df[
            self.transaction_df["t_dat"] >= self.train_threshold
        ]
        self.test_customer_ids = self.test_df["customer_id"].unique()

    def load_train_df(self):
        print("transaction data fast load")
        with open(os.path.join(self.data_path, "train_df.pickle"), "rb") as handle:
            self.train_df = pickle.load(handle)

    def load_test_df(self):
        print("transaction data fast load")
        with open(os.path.join(self.data_path, "test_df.pickle"), "rb") as handle:
            self.test_df = pickle.load(handle)

    def get_customer_article_count(self):
        print("calculating customer and article counts")
        self.customer_ids = self.customer_df["customer_id"].unique()
        self.article_ids = self.article_df["article_id"].unique()
        self.customer_count = len(self.customer_ids)
        self.article_count = len(self.article_ids)

    def load_S_data(self):
        print("S_train fast load")
        with open(os.path.join(self.data_path, "S_train.pickle"), "rb") as handle:
            self.S_train = pickle.load(handle)

        print("S_test fast load")
        with open(os.path.join(self.data_path, "S_test.pickle"), "rb") as handle:
            self.S_test = pickle.load(handle)

    def get_test_gts(self):
        print("get_test_gt fast load")
        with open(os.path.join(self.data_path, "test_gts.pickle"), "rb") as handle:
            self.test_gts = pickle.load(handle)

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


#%%
if __name__ == "__main__":
    # from params import OG_DATA_PATH, PROCESSED_DATA_PATH, PROCESSED_DATA_OUT_PATH

    dataprocessor = DataLoader()
    #%%
