import os

POSSIBLE_DATASETS = ["full", "only_test_customers", "only_test", "small"]


# TODO cleanup
def get_params(ds=POSSIBLE_DATASETS[0]):
    BASE_PATH = "/home/clement/Desktop/projects/personalized-recommendations/"
    OG_DATA_NAME = "h-and-m-personalized-fashion-recommendations"
    OG_PATH = os.path.join(BASE_PATH, OG_DATA_NAME)
    OUT_PATH = os.path.join(BASE_PATH, "out")
    OUT_DS_PATH = os.path.join(OUT_PATH, ds)
    PROCESSED_DATA_OUT_PATH = os.path.join(OUT_DS_PATH, "data")
    FIGURES_PATH = os.path.join(OUT_DS_PATH, "figures")
    MODEL_LOAD_PATH = os.path.join(OUT_DS_PATH, "models")
    MODEL_SAVE_PATH = os.path.join(OUT_DS_PATH, "models")

    params = {
        "predict_amount": 12,
        "model_name": "base",
        "lr": 2,
        "factor_num": 32,
        "weight_decay": 1e-5,
        "end_epoch": 10,
        "ds": ds,
        "og_path": OG_PATH,
        "out_path": OUT_PATH,
        "p_out_path": PROCESSED_DATA_OUT_PATH,
        "figures_path": FIGURES_PATH,
        "model_load_path": MODEL_LOAD_PATH,
        "model_save_path": MODEL_SAVE_PATH,
        "base_path": BASE_PATH,
        "data_path": PROCESSED_DATA_OUT_PATH,
        "batch_size": 64
    }
    return params