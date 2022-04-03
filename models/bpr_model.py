# -*- coding: utf-8 -*-

import torch
import numpy as np
from utils.evaluate_map_score import evaluate_map_score
import math
import time
import os
from utils.dataloader import DataLoader
from utils.rmse import RMSE
from models.nn_model import NNModel
from torch.utils.tensorboard import SummaryWriter


class BPRModel:
    def __init__(self, params):
        self.set_model_version()
        self.params = params
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Device is {self.device}")
        self.setup_dataloader()
        self.setup_network()
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
        )
        self.loss_function = RMSE()

        self.start_epoch = 1
        self.best_score = 0

        self.writer = SummaryWriter(
            log_dir=os.path.join(self.params["tb_path"], self.params_to_str())
        )

        if os.path.exists(
            os.path.join(self.params["model_load_path"], self.params_to_str())
        ):
            print(f"loading model {self.params['model_name']}")
            checkpoint = torch.load(
                os.path.join(self.params["model_load_path"], self.params_to_str(),),
                map_location=self.device,
            )
            self.net.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epoch"]
            self.best_score = checkpoint["score"]
            print("start epoch ", self.start_epoch, ", best score ", self.best_score)

    def evaluate_BPR(self, customer_ids, gt):
        pred = self.recommend_BPR(customer_ids)
        map_score = evaluate_map_score(pred, gt, self.params["predict_amount"])
        return map_score

    def set_model_version(self):
        self.model_version = "BPRModel-1.0.0"

    def params_to_str(self):
        return (
            f"{self.model_version}"
            + f"{self.params['model_name']}"
            + f"|{self.params['lr']}"
            + f"|{self.params['factor_num']}"
            + f"|{self.params['weight_decay']}"
            + f"|{self.params['batch_size']}"
        )

    def params_to_dict(self):  # TODO remove
        pass

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

    def setup_train(self):
        self.train_loss_history = []
        self.train_map_history = []
        self.test_map_history = []
        self.load_new_dataloader()

    def setup_epoch(self):
        self.start_time = time.time()
        self.total_loss = 0
        self.total_i = 0
        self.total_j = 0

    def run_batch(self, batch):
        batch = batch.to(self.device)
        customer, article_i, article_j = batch[:, 0], batch[:, 1], batch[:, 2]
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

    def load_new_dataloader(self):
        self.train_loader = self.dataloader.get_new_loader(
            batch_size=self.params["batch_size"]
        )

    def end_epoch(self, epoch):
        print("\n")
        print(
            f"Epoch {epoch}, loss {self.total_loss/len(self.train_loader):.5f}"
            + f" | time {int(time.time()-self.start_time)}"
            + f" | total_i {self.total_i/len(self.train_loader)}"
            + f" | total_j {self.total_j/len(self.train_loader)}"
        )

        self.writer.add_scalar(
            "Loss/train", self.total_loss / len(self.train_loader), epoch
        )
        self.train_loss_history.append(self.total_loss / len(self.train_loader))
        if epoch % self.params["dataloader_per_epoch"] == 0:
            self.load_new_dataloader()

        if epoch % self.params["evaluate_per_epoch"] == 0:
            self.net.eval()
            with torch.no_grad():
                train_score = self.evaluate_BPR(
                    customer_ids=self.dataloader.train_customer_ids,
                    gt=self.dataloader.train_customer_articles.values,
                )
                self.test_map_history.append(train_score)
                self.writer.add_scalar("Map Score/train", train_score, epoch)

                test_score = self.evaluate_BPR(
                    customer_ids=self.dataloader.test_customer_ids,
                    gt=self.dataloader.test_customer_articles.values,
                )
                self.writer.add_scalar("Map Score/test", test_score, epoch)
                self.test_map_history.append(test_score)
            print(f"train_score is {train_score:.10f}")
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
                    os.path.join(self.params["model_save_path"], self.params_to_str()),
                )

    def train(self):
        self.setup_train()
        for epoch in range(self.start_epoch, self.params["end_epoch"]):
            self.setup_epoch()
            self.net.train()

            for i, batch in enumerate(self.train_loader):
                print(f"\rBatch number {i} | {len(self.train_loader)}", end="")
                batch = batch.to(self.device)
                self.run_batch(batch)

            self.end_epoch(epoch)

    def setup_dataloader(self):
        self.dataloader = DataLoader(self.params["data_path"])

    def setup_network(self):
        self.net = NNModel(
            self.dataloader.customer_count,
            self.dataloader.article_count,
            self.params["factor_num"],
            self.device,
        )


if __name__ == "__main__":
    from params import bpr_params

    model = BPRModel(bpr_params)
    model.train()
