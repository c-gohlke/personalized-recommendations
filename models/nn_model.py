import torch
from utils.rmse import RMSE


class NNModel(torch.nn.Module):
    def __init__(self, customer_count, article_count, factor_num, device):
        super().__init__()
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
        self.loss_function = RMSE()

    def forward(self, customer, article_i, article_j):
        customer = self.embed_customer(customer)
        article_i = self.embed_article(article_i)
        article_j = self.embed_article(article_j)

        prediction_i = (customer * article_i).sum(1)
        prediction_j = (customer * article_j).sum(1)

        return prediction_i.sigmoid(), prediction_j.sigmoid()
