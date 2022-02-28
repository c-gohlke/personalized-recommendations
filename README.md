## Setup

conda create --name h-and-m --file conda-env.txt
python -m utils.dataprocessor
run python.ipynb

## Data exploration

### articles

There are 105,542 total articles. (id 0 to 105,539)
Each article has a corresponding image, meaning image similarity is a metric that can be used.

Each article has a product code which is not unique (same article in different colors?), product name (similarity pred?), product type name (jean, sweater, top, Hair clip, Umbrella), product group name (Garment Upper Body, Lower Body, accessory, night wear), appearance (solid, stripe, transparent) and colour group (white, light grey, dark blue etc.), price(regularized different currencies)


### customers

There are 1,371,980 total customers. (id 0 to 1,371,979)
Customers have club_member_status (Active 93%, preregistered 7%), fashion news frequency (None 64% or regular 35%), age (Important?, peak 21, left-tailed) and postal code (encrypted, 1.2 million values - clustering similar postal codes probably impossible), Fashion News accepted boolean and boolean if customer active for communications.

### transactions

There are 31,788,324 total transactions.

online/offline channel, price, transaction date.

The least amount of transactions for a customer is 1, and the most is 1895.
The mean amount of transactions per customer is 23.33, the median is 9.

The min transactions for a article is 1, the max transactions is 50287
Average amount of transactions per article is 304, median amount is 65

Amount of times customer buys same article: max amount of same article bought is X (32),
mean times same article bought per customer is X

Amount of transactions per product group name
Amount of transactions per colour group
Amount of transactions per product type name

### test set

transactions made in the last 7 days

### Task

predict 12 items to recommend for a given nuser, that can have past buying history or not.