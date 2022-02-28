import torch

from models.model import Model
from params import run_params

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA Unavailable")
        run_params["device"] = "cpu"

    model = Model(run_params=run_params)
    model.load_model()
    model.train()
    model.evaluate()
