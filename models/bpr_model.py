# -*- coding: utf-8 -*-

import torch
import numpy as np
from utils.map_score import map_score
from params import PREDICT_AMOUNT, DEVICE
import math


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

    def forward(self, customer, article_i, article_j):
        customer = self.embed_customer(customer)
        article_i = self.embed_article(article_i)
        article_j = self.embed_article(article_j)

        prediction_i = torch.empty((len(customer), 1))
        prediction_j = torch.empty((len(customer), 1))

        for b in range(len(customer)):
            prediction_i[b] = torch.dot(customer[b], article_i[b])
            prediction_j[b] = torch.dot(customer[b], article_j[b])

        return prediction_i.sigmoid(), prediction_j.sigmoid()

    def evaluate_BPR(self, test_customer_ids, test_gts):
        pred = self.recommend_BPR(test_customer_ids)
        score = map_score(pred, test_gts)
        return score

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


# class BPR_Model(torch.nn.Module):
#     def __init__(self, customer_count, article_count, factor_num):
#         super(BPR_Model, self).__init__()

#         """
# 		user_num: number of users;
# 		item_num: number of items;
# 		factor_num: number of predictive factors.
# 		"""
#         self.device = torch.device(DEVICE)

#         self.embed_customer = torch.nn.Embedding(
#             customer_count, factor_num, device=self.device
#         )
#         self.embed_article = torch.nn.Embedding(
#             article_count, factor_num, device=self.device
#         )
#         self.article_count = article_count
#         # self.embed_customer.to(self.device)
#         # self.embed_article.to(self.device)

#         torch.nn.init.normal_(self.embed_customer.weight, std=0.01)
#         torch.nn.init.normal_(self.embed_article.weight, std=0.01)

#     def forward(self, customer, article_i, article_j):
#         customer = self.embed_customer(customer)
#         article_i = self.embed_article(article_i)
#         article_j = self.embed_article(article_j)

#         prediction_i = (customer * article_i).sum(dim=-1)
#         prediction_j = (customer * article_j).sum(dim=-1)
#         return prediction_i, prediction_j

#     def evaluate_BPR(self, test_customer_ids, test_gts):
#         pred = self.recommend_BPR(test_customer_ids)
#         score = map_score(pred, test_gts)
#         return score

#     def recommend_BPR(self, customer_ids):
#         item_i = torch.tensor(range(self.article_count)).to(self.device).reshape(-1)
#         e_item = self.embed_article(item_i)

#         recommendations = np.empty((len(customer_ids), PREDICT_AMOUNT))
#         if DEVICE == "cuda" or "cuda:0":
#             # TODO check if indeed faster
#             batch_size = 1000
#         else:
#             batch_size = 10000
#         for i in range(math.ceil(len(customer_ids) / batch_size)):
#             start_i = batch_size * i
#             end_i = min(batch_size * (i + 1), len(customer_ids))
#             batch = customer_ids[start_i:end_i]
#             cid = torch.tensor(batch).to(self.device).reshape(-1)

#             e_customer = self.embed_customer(cid)
#             score = torch.matmul(e_customer, e_item.t())
#             _, indices = torch.topk(score, PREDICT_AMOUNT)
#             b_rec = torch.take(item_i, indices).cpu().numpy()
#             recommendations[start_i:end_i, :] = b_rec

#         return recommendations
