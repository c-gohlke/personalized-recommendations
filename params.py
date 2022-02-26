import os

POSSIBLE_DATASETS = ["full", "only_test_customers", "only_test", "small"]
DS = POSSIBLE_DATASETS[0]

DEVICE = "cuda:0"
# DEVICE = "cpu"

BASE_PATH = "/home/clement/Desktop/projects/personalized-recommendations/"
OG_DATA_NAME = "h-and-m-personalized-fashion-recommendations"
OG_DATA_PATH = os.path.join(BASE_PATH, OG_DATA_NAME)
OUT_PATH = os.path.join(BASE_PATH, "out", DS)
PROCESSED_DATA_PATH = os.path.join(OUT_PATH, "data")
PROCESSED_DATA_OUT_PATH = os.path.join(OUT_PATH, "data")
FIGURES_PATH = os.path.join(OUT_PATH, "figures")
MODEL_LOAD_PATH = os.path.join(OUT_PATH, "models")
MODEL_SAVE_PATH = os.path.join(OUT_PATH, "models")

PREDICT_AMOUNT = 12

# model_params = {"in0": 10, "out0": 10, "in1": 10, "out1": 10, "device": "cuda"}

# run_params = {
#     "lr": 1e-3,
#     "start_epoch": 0,
#     "end_epoch": 3,
#     "loss": "mse",
#     "print_per_epoch": 1,
#     "save_per_epoch": 1,
#     "b_size": 32,
#     "save_name": "Base-1",
#     "load_name": "Base-1",
#     "device": "cuda",
#     "incompatible": [
#         "loss",
#         "model_params",
#     ],  # value must be the same to load successfully
#     "model_params": model_params,
# }
