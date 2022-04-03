import torch
from utils.rmse import RMSE


class NNProfileModel(torch.nn.Module):
    def __init__(
        self, article_count, factor_num, customer_profile_count, dropout_rate, device
    ):
        super().__init__()
        """
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
		"""
        self.device = device
        self.article_count = article_count
        self.factor_num = factor_num
        self.customer_profile_count = customer_profile_count
        self.dropout_rate = dropout_rate

        self.embed_article = torch.nn.Embedding(
            article_count + 1, factor_num, device=self.device
        )  # 1 value for encoding of "no article" used for profiling customer form articles

        self.linear_c = torch.nn.Linear(
            customer_profile_count * factor_num, factor_num, device=self.device
        )
        self.norm_c = torch.nn.BatchNorm1d(factor_num, device=self.device)
        self.activate_c = torch.nn.PReLU(device=self.device)
        self.dropout_c = torch.nn.Dropout(dropout_rate)

        torch.nn.init.normal_(self.embed_article.weight, std=0.01)
        torch.nn.init.normal_(self.linear_c.weight, std=0.01)
        self.loss_function = RMSE()

        self.output_layer_bias = torch.nn.Parameter(torch.Tensor(article_count + 1)).to(
            self.device
        )
        self.output_layer_bias.data.normal_(0.0, 0.01)

    def forward(self, customer, article_i, article_j=None):
        customer_profile = self.embed_article(customer).reshape(
            -1, self.customer_profile_count * self.factor_num
        )
        customer_profile = self.linear_c(customer_profile)
        customer_profile = self.norm_c(customer_profile)
        customer_profile = self.activate_c(customer_profile)
        customer_profile = self.dropout_c(customer_profile)

        article_i = self.embed_article(article_i)
        prediction_i = torch.matmul(customer_profile, article_i.T)

        logits = torch.nn.functional.linear(
            customer_profile, self.embed_article.weight, bias=self.output_layer_bias
        )

        if not article_j is None:
            article_j = self.embed_article(article_j)
            prediction_j = torch.matmul(customer_profile, article_j.T)
            return prediction_i.sigmoid(), prediction_j.sigmoid()
        else:
            return prediction_i.sigmoid()
