import pandas as pd
import os
import numpy as np
from params import OG_DATA_PATH


def explore_articles():
    articles_df = pd.read_csv(os.path.join(OG_DATA_PATH, "articles.csv"))

    # how many articles?
    print(len(articles_df))

    # features?
    print("article columns", articles_df.columns)


def explore_customers():
    customers_df = pd.read_csv(os.path.join(OG_DATA_PATH, "customers.csv"))

    # how many customers?
    print(len(customers_df))

    # features?
    print("customer columns", customers_df.columns)


def explore_transactions():
    # customers_df = pd.read_csv(os.path.join(OG_DATA_PATH, 'customers.csv'))
    train_df = pd.read_csv(os.path.join(OG_DATA_PATH, "transactions_train.csv"))

    # how many transactions?
    # print(len(train_df))

    # features?
    print("transaction columns", train_df.columns)

    # How many articles per customer?
    # print("a/customer", len(train_df)/len(customers_df))

    # min max amount transactions per customer?
    # customer_transactions = train_df.groupby("customer_id").count()
    # print(customer_transactions)
    # print(type(customer_transactions))
    # print(customer_transactions.min())
    # print(customer_transactions.max())
    # print(customer_transactions.mean())
    # print(customer_transactions.median())

    # min max amount transactions per articles?
    # article_transactions = train_df.groupby("article_id").count()
    # print(article_transactions)
    # print(type(article_transactions))

    # print(article_transactions.min())
    # print(article_transactions.max())
    # print(article_transactions.mean())
    # print(article_transactions.median())

    max_c = 0
    counts = []
    customer_ids = train_df["customer_id"].unique()
    for customer in customer_ids:
        customer_df = train_df[train_df["customer_id"] == customer]
        customer_article_transactions = (
            customer_df.groupby(by=["article_id"])["article_id"].value_counts().values
        )

        # print("min", customer_article_transactions.min())
        print("max", customer_article_transactions.max())
        max_transactions = customer_article_transactions.max()
        if max_transactions > max_c:
            max_c = max_transactions
        counts.append(customer_article_transactions.mean())
    print("max of max", max_c)
    print("mean", np.mean(np.array(counts)))


# distribution of n articles and how often bought

if __name__ == "__main__":
    explore_transactions()
