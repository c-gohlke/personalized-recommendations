import os
import numpy as np
import torch
import pickle
from utils.dataset import Dataset
import pandas as pd
import random


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_S_train()
        self.load_test_df()
        self.load_meta_data()
        self.load_train_df()

    # def load_customer_df(self):
    #     print("customer data fast load")
    #     with open(os.path.join(self.data_path, "customer_df.pickle"), "rb") as handle:
    #         self.customer_df = pickle.load(handle)

    # def load_article_df(self):
    #     print("article data fast load")
    #     with open(os.path.join(self.data_path, "article_df.pickle"), "rb") as handle:
    #         self.article_df = pickle.load(handle)

    def load_train_df(self):
        print("load train data")
        with open(os.path.join(self.data_path, "train_df.pickle"), "rb") as handle:
            self.train_df = pickle.load(handle)

        with open(
            os.path.join(self.data_path, "train_customer_articles.pickle"), "rb"
        ) as handle:
            self.train_customer_articles = pickle.load(handle)

    def load_test_df(self):
        print("transaction data fast load")
        with open(os.path.join(self.data_path, "test_df.pickle"), "rb") as handle:
            self.test_df = pickle.load(handle)
        with open(
            os.path.join(self.data_path, "test_customer_articles.pickle"), "rb"
        ) as handle:
            self.test_customer_articles = pickle.load(handle)

    def load_meta_data(self):
        # with open(os.path.join(self.data_path, "customer_ids.pickle"), "rb") as handle:
        #     self.customer_ids = pickle.load(handle)
        # with open(os.path.join(self.data_path, "article_ids.pickle"), "rb") as handle:
        #     self.article_ids = pickle.load(handle)
        with open(
            os.path.join(self.data_path, "customer_count.pickle"), "rb"
        ) as handle:
            self.customer_count = pickle.load(handle)
        with open(os.path.join(self.data_path, "article_count.pickle"), "rb") as handle:
            self.article_count = pickle.load(handle)
        with open(
            os.path.join(self.data_path, "train_customer_ids.pickle"), "rb"
        ) as handle:
            self.train_customer_ids = pickle.load(handle)
        with open(
            os.path.join(self.data_path, "test_customer_ids.pickle"), "rb"
        ) as handle:
            self.test_customer_ids = pickle.load(handle)

    def load_S_train(self):
        print("S_train fast load")
        with open(os.path.join(self.data_path, "S_train.pickle"), "rb") as handle:
            self.S_train = pickle.load(handle)

    # def load_S_test(self):
    #     print("S_test fast load")
    #     with open(os.path.join(self.data_path, "S_test.pickle"), "rb") as handle:
    #         self.S_test = pickle.load(handle)

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

        ones = self.train_customer_articles.apply(lambda x: np.random.choice(x))
        X[:, 1] = ones

        zeros = np.random.randint(self.article_count, size=len(self.train_customer_ids))
        drawn_index = range(len(zeros))
        check_vals = pd.Series(drawn_index)
        redraw_bool = check_vals.apply(
            lambda x: zeros[x] in self.train_customer_articles.values[x]
        )
        redraw_index = redraw_bool[redraw_bool].index
        while len(redraw_index) > 0:
            zeros[redraw_index] = np.random.randint(
                self.article_count, size=len(redraw_index)
            )
            check_vals = pd.Series(redraw_index)
            redraw_bool = check_vals.apply(
                lambda x: zeros[x] in self.train_customer_articles.values[x]
            )
            redraw_index = redraw_bool[redraw_bool].index
        X[:, 2] = zeros

        return X


if __name__ == "__main__":
    from params import params

    dataloader = DataLoader(params["data_path"])
