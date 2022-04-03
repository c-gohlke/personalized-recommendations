# -*- coding: utf-8 -*-

import torch
import numpy as np
import math
import pandas as pd

from utils.dataloader_profile import DataLoaderProfile
from utils.evaluate_map_score import evaluate_map_score

from models.bpr_model import BPRModel
from models.nn_profile_model import NNProfileModel


def get_c_profile(c, c_articles, noneval, customer_profile_count):
    if c in c_articles.index:
        c_possible_articles = c_articles[c]
        if len(c_possible_articles) <= customer_profile_count:
            return np.concatenate(
                (
                    [
                        noneval
                        for _ in range(
                            customer_profile_count - len(c_possible_articles)
                        )
                    ],
                    c_possible_articles,
                )
            ).astype(np.int32)
        else:
            return np.random.choice(
                c_possible_articles, customer_profile_count, replace=False
            )
    else:
        return [noneval for _ in range(customer_profile_count)]


class BPRProfileModel(BPRModel):
    def __init__(self, params):
        super().__init__(params)

    def set_model_version(self):
        self.model_version = "BPRProfileModel-1.0.0"

    def customer_profile(self, cid):
        cid = pd.Series(cid)
        noneval = self.dataloader.article_count
        c_profile = cid.apply(
            lambda x: get_c_profile(
                x,
                self.dataloader.train_customer_articles,
                noneval,
                self.params["customer_profile_count"],
            )
        ).to_numpy()
        c_profile = np.concatenate((c_profile)).reshape(
            -1, self.params["customer_profile_count"]
        )
        return torch.Tensor(c_profile).int().to(self.device)

    def load_new_dataloader(self):
        self.train_loader = self.dataloader.get_new_loader(
            batch_size=self.params["batch_size"],
            customer_profile_count=self.params["customer_profile_count"],
        )

    def setup_network(self):
        self.net = NNProfileModel(
            self.dataloader.article_count,
            self.params["factor_num"],
            self.params["customer_profile_count"],
            self.params["dropout_rate"],
            self.device,
        )

    def setup_dataloader(self):
        self.dataloader = DataLoaderProfile(self.params["data_path"])

    def recommend_BPR(self, customer_ids):
        item_i = (
            torch.tensor(range(self.dataloader.article_count))
            .to(self.device)
            .reshape(-1)
        )

        recommendations = np.empty((len(customer_ids), self.params["predict_amount"]))
        if self.device.type == "cuda":
            batch_size = 1500  # about 1GB
        else:
            batch_size = 100000
        for i in range(math.ceil(len(customer_ids) / batch_size)):
            start_i = batch_size * i
            end_i = min(batch_size * (i + 1), len(customer_ids))
            batch_cid = customer_ids[start_i:end_i]
            # print(batch_cid)
            customer_profile = self.customer_profile(batch_cid)
            score = self.net(customer_profile, item_i)
            # print(score)
            _, indices = torch.topk(score, self.params["predict_amount"])
            b_rec = torch.take(item_i, indices).cpu().numpy()
            recommendations[start_i:end_i, :] = b_rec

        return recommendations

    def run_batch(self, batch):
        batch = batch.to(self.device)
        customer, article_i, article_j = (
            batch[:, :-2],
            batch[:, -2],
            batch[:, -1],
        )

        self.net.zero_grad()
        prediction_i, prediction_j = self.net(customer, article_i, article_j)

        i_loss = (prediction_i - 1).pow(2).sum().sqrt()  # i_target = 1
        j_loss = prediction_j.pow(2).sum().sqrt()  # j_target = 0
        loss = i_loss + j_loss

        loss.backward()
        self.optimizer.step()
        self.total_loss = self.total_loss + loss.item() / (len(batch) * 2)
        self.total_i = self.total_i + prediction_i.mean().item()
        self.total_j = self.total_j + prediction_j.mean().item()


if __name__ == "__main__":
    from params import bpr_profile_params

    model = BPRProfileModel(bpr_profile_params)
    model.train()
