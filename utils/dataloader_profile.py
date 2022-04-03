import os
import numpy as np
import torch
import pickle
from utils.dataset import Dataset
import pandas as pd
import random
from utils.dataloader import DataLoader


class DataLoaderProfile(DataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)

    def get_new_loader(self, batch_size, customer_profile_count):
        train_data = self.sample_train(customer_profile_count)
        train_data = torch.Tensor(train_data).int()
        train_set = Dataset(train_data)
        return torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, pin_memory=True,
        )

    def sample_train(self, customer_profile_count):
        X = np.empty((len(self.train_customer_ids), customer_profile_count + 2)).astype(
            np.int32
        )

        # X[:, :-2] is customer profile, X[:, -2] is a true article, X[:, -1] is a false article
        profile_one = self.train_customer_articles.apply(
            lambda x: pick_profiles(
                x, customer_profile_count, noneval=self.article_count
            )
        )
        X[:, :-1] = np.concatenate((profile_one.values)).reshape(
            -1, customer_profile_count + 1
        )

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
        X[:, -1] = zeros

        return X


def pick_profiles(x, customer_profile_count, noneval):
    # TODO cleaner code
    if len(x) <= customer_profile_count + 1:
        temp = np.concatenate(
            ([noneval for _ in range(customer_profile_count + 1 - len(x))], x)
        ).astype(np.int32)
    else:
        temp = np.random.choice(x, customer_profile_count + 1, replace=False)
    return temp


if __name__ == "__main__":
    from params import params

    dataloader = DataLoaderProfile(params["data_path"])
