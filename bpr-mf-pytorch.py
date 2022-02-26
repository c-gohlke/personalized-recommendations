#%% import libraries, set params

import torch
import time
import os

from models.bpr_model import BPR_Model

# from utils.sampler import sampler
from utils.dataprocessor import DataProcessor

from params import MODEL_LOAD_PATH, MODEL_SAVE_PATH, DS

factor_num = 32
model_name = "lr0.5" + str(factor_num)
lr = 0.5
weight_decay = 0
start_epoch = 1
batch_size = 64
BEST_SCORE = 0

if not torch.cuda.is_available():
    print("CUDA UNAVAILABLE")


#%% start dataprocessors

dataprocessor = DataProcessor()

#%% init model

model = BPR_Model(dataprocessor.customer_count, dataprocessor.article_count, factor_num)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

if os.path.exists(os.path.join(MODEL_LOAD_PATH, f"{model_name}_BPR.pt")):
    print(f"loading model {model_name}")
    checkpoint = torch.load(
        os.path.join(MODEL_LOAD_PATH, f"{model_name}_BPR.pt"), map_location=model.device
    )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    BEST_SCORE = checkpoint["score"]

#%% build dataloader

train_loader = dataprocessor.get_new_loader(batch_size=batch_size)
print("dataloader built")

#%% train

train_history = []
test_history = []

evaluate_per_epoch = 5
dataloader_per_epoch = 10
end_epoch = 200

print(start_epoch, end_epoch)
for epoch in range(start_epoch, end_epoch):
    print(f"Epoch {epoch}")
    model.train()
    start_time = time.time()
    total_loss = 0

    # train_loader = dataprocessor.get_new_loader(batch_size=batch_size)
    # print("dataloader built")
    total_i = 0
    total_j = 0

    for i, batch in enumerate(train_loader):
        print(f"\rBatch number {i} | {len(train_loader)}", end="")
        batch = batch.to(model.device)

        customer, article_i, article_j = batch[:, 0], batch[:, 1], batch[:, 2]
        model.zero_grad()
        prediction_i, prediction_j = model(customer, article_i, article_j)

        i_loss = (prediction_i - 1).pow(2).sum().sqrt()  # i_target = 1
        j_loss = prediction_j.pow(2).sum().sqrt()  # j_target = 0
        loss = i_loss + j_loss

        loss.backward()
        optimizer.step()
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
    if epoch % dataloader_per_epoch == 0:
        print("building new dataloader")
        train_loader = dataprocessor.get_new_loader(batch_size=batch_size)
        print("dataloader built")

    if epoch % evaluate_per_epoch == 0:
        model.eval()
        with torch.no_grad():
            test_score = model.evaluate_BPR(
                dataprocessor.test_customer_ids, dataprocessor.test_gts
            )

            test_history.append(test_score)
        print(f"test_score is {test_score}")  # ":.10f}")

        if test_score > BEST_SCORE:
            BEST_SCORE = test_score
            print(f"******new best score for DS {DS} is {BEST_SCORE}")
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH)

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "score": test_score,
                },
                os.path.join(MODEL_SAVE_PATH, f"{model_name}_BPR.pt"),
            )

#%% plot
# only test 50 epochs 0.0007217210909614171 lr0.05 w_decay0

# torch.save(
#     {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),},
#     os.path.join(MODEL_SAVE_PATH, f"{model_name}_BPR.pt"),
# )
