# -*- coding: utf-8 -*-

import torch
import numpy as np
from utils.map_score import map_score
import math
import time
import os
from utils.dataloader import DataLoader


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-15):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class NN_Model(torch.nn.Module):
    def __init__(self, customer_count, article_count, factor_num, device):
        super(NN_Model, self).__init__()
        """
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
		"""
        self.device = device
        self.embed_customer = torch.nn.Embedding(
            customer_count, factor_num, device=self.device
        )
        self.embed_article = torch.nn.Embedding(
            article_count, factor_num, device=self.device
        )

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


class BPR_Model:
    def __init__(self, params):
        self.params = params
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Device is {self.device}")
        self.dataloader = DataLoader(self.params["data_path"])
        self.net = NN_Model(
            self.dataloader.customer_count,
            self.dataloader.article_count,
            self.params["factor_num"],
            self.device,
        )
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
        )

        self.start_epoch = 0
        self.best_score = 0

        if os.path.exists(
            os.path.join(
                self.params["model_load_path"], f"{self.params['model_name']}_BPR.pt"
            )
        ):
            print(f"loading model {self.params['model_name']}")
            checkpoint = torch.load(
                os.path.join(
                    self.params["model_load_path"],
                    f"{self.params['model_name']}_BPR.pt",
                ),
                map_location=self.device,
            )
            self.net.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epoch"]
            self.best_score = checkpoint["score"]

    def evaluate_BPR(self, test_customer_ids, test_gts):
        pred = self.recommend_BPR(test_customer_ids)
        score = map_score(pred, test_gts, self.params["predict_amount"])
        return score

    def loss(self, pred_i, pred_j):
        gt_i = torch.ones(len(pred_i))
        gt_j = torch.zeros(len(pred_j))

        return self.loss_function(pred_i, gt_i) + self.loss_function(pred_j, gt_j)

    def recommend_BPR(self, customer_ids):
        item_i = (
            torch.tensor(range(self.dataloader.article_count))
            .to(self.device)
            .reshape(-1)
        )
        e_item = self.net.embed_article(item_i)

        recommendations = np.empty((len(customer_ids), self.params["predict_amount"]))
        if self.device.type == "cuda":
            # TODO check if indeed faster
            batch_size = 1000
        else:
            batch_size = 10000
        for i in range(math.ceil(len(customer_ids) / batch_size)):
            start_i = batch_size * i
            end_i = min(batch_size * (i + 1), len(customer_ids))
            batch = customer_ids[start_i:end_i]
            cid = torch.tensor(batch).to(self.device).reshape(-1)

            e_customer = self.net.embed_customer(cid)
            score = torch.matmul(e_customer, e_item.t())
            _, indices = torch.topk(score, self.params["predict_amount"])
            b_rec = torch.take(item_i, indices).cpu().numpy()
            recommendations[start_i:end_i, :] = b_rec

        return recommendations

    def train(self):
        train_history = []
        test_history = []

        print("getting train_loader")
        train_loader = self.dataloader.get_new_loader(
            batch_size=self.params["batch_size"]
        )

        for epoch in range(self.start_epoch, self.params["end_epoch"]):
            print(f"Epoch {epoch}")
            self.net.train()
            start_time = time.time()
            total_loss = 0

            total_i = 0
            total_j = 0

            for i, batch in enumerate(train_loader):
                print(f"\rBatch number {i} | {len(train_loader)}", end="")
                batch = batch.to(self.device)

                customer, article_i, article_j = batch[:, 0], batch[:, 1], batch[:, 2]
                self.net.zero_grad()
                prediction_i, prediction_j = self.net(customer, article_i, article_j)

                i_loss = (prediction_i - 1).pow(2).sum().sqrt()  # i_target = 1
                j_loss = prediction_j.pow(2).sum().sqrt()  # j_target = 0
                loss = i_loss + j_loss

                loss.backward()
                self.optimizer.step()
                total_loss = total_loss + loss.item() / (len(batch) * 2)
                total_i = total_i + prediction_i.mean().item()
                total_j = total_j + prediction_j.mean().item()

            print("\n")
            print(
                f"Epoch {epoch}, loss {total_loss/len(train_loader)}"  # ":.10f},
                + f" | time {int(time.time()-start_time)}"
                + f" | total_i {total_i/len(train_loader)}"
                + f" | total_j {total_j/len(train_loader)}"
            )

            train_history.append(total_loss / len(train_loader))
            if epoch % self.params["dataloader_per_epoch"] == 0:
                print("building new dataloader")
                train_loader = self.dataloader.get_new_loader(
                    batch_size=self.params["batch_size"]
                )
                print("dataloader built")

            if epoch % self.params["evaluate_per_epoch"] == 0:
                self.net.eval()
                with torch.no_grad():
                    test_score = self.evaluate_BPR(
                        self.dataloader.test_customer_ids, self.dataloader.test_gt
                    )

                    test_history.append(test_score)
                print(f"test_score is {test_score:.10f}")

                if test_score > self.best_score:
                    self.best_score = test_score
                    print(
                        f"*new best score for DS {self.params['ds']} is {self.best_score}"
                    )
                    if not os.path.exists(self.params["model_save_path"]):
                        os.makedirs(self.params["model_save_path"])

                    torch.save(
                        {
                            "epoch": epoch,
                            "model": self.net.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "score": self.best_score,
                        },
                        os.path.join(
                            self.params["model_save_path"],
                            f"{self.params['model_name']}_BPR.pt",
                        ),
                    )
