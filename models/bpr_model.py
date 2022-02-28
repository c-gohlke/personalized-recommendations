# -*- coding: utf-8 -*-

import torch
import numpy as np
from utils.map_score import map_score
from params import PREDICT_AMOUNT, DEVICE
import math


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-15):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class BPR_Model(torch.nn.Module):
    def __init__(self, customer_count, article_count, factor_num):
        super(BPR_Model, self).__init__()

        """
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
		"""
        self.device = torch.device(DEVICE)

        self.embed_customer = torch.nn.Embedding(
            customer_count, factor_num, device=self.device
        )
        self.embed_article = torch.nn.Embedding(
            article_count, factor_num, device=self.device
        )
        self.article_count = article_count
        # self.embed_customer.to(self.device)
        # self.embed_article.to(self.device)

        torch.nn.init.normal_(self.embed_customer.weight, std=0.01)
        torch.nn.init.normal_(self.embed_article.weight, std=0.01)
        self.loss_function = RMSELoss()

    def forward(self, customer, article_i, article_j):
        customer = self.embed_customer(customer)
        article_i = self.embed_article(article_i)
        article_j = self.embed_article(article_j)

        prediction_i = (customer * article_i).sum(1)
        prediction_j = (customer * article_j).sum(1)

        return prediction_i.sigmoid(), prediction_j.sigmoid()

    def evaluate_BPR(self, test_customer_ids, test_gts):
        pred = self.recommend_BPR(test_customer_ids)
        score = map_score(pred, test_gts)
        return score

    def loss(self, pred_i, pred_j):
        gt_i = torch.ones(len(pred_i))
        gt_j = torch.zeros(len(pred_j))

        return self.loss_function(pred_i, gt_i) + self.loss_function(pred_j, gt_j)

    def recommend_BPR(self, customer_ids):
        item_i = torch.tensor(range(self.article_count)).to(self.device).reshape(-1)
        e_item = self.embed_article(item_i)

        recommendations = np.empty((len(customer_ids), PREDICT_AMOUNT))
        if DEVICE == "cuda" or "cuda:0":
            # TODO check if indeed faster
            batch_size = 1000
        else:
            batch_size = 10000
        for i in range(math.ceil(len(customer_ids) / batch_size)):
            start_i = batch_size * i
            end_i = min(batch_size * (i + 1), len(customer_ids))
            batch = customer_ids[start_i:end_i]
            cid = torch.tensor(batch).to(self.device).reshape(-1)

            e_customer = self.embed_customer(cid)
            score = torch.matmul(e_customer, e_item.t())
            _, indices = torch.topk(score, PREDICT_AMOUNT)
            b_rec = torch.take(item_i, indices).cpu().numpy()
            recommendations[start_i:end_i, :] = b_rec

        return recommendations
