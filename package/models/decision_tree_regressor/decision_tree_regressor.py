import os
import sys
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import random
import json

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(currentdir))
from model_base import ModelBase

import logging

SEED = 42
GROUP_NAME = "DecisionTreeRegressor"
MODEL_NAME = "2.0"

np.random.seed(SEED)
random.seed(SEED)

os.environ["WANDB_MODE"] = "dryrun"


logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-10s][%(funcName)-10s][%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename="data/log/models/decision_tree_regressor.log"),
    ],
)


class ModelDecisionTreeRegressor(ModelBase):
    def __init__(self, model_folder: str):
        super().__init__(
            group_name=GROUP_NAME, model_name=MODEL_NAME, model_folder=model_folder
        )

    def load_grid(self):
        current_path = os.path.dirname(__file__)
        grid_path = os.path.join(current_path, "grid.json")
        with open(grid_path) as json_file:
            self.gs_params = json.load(json_file)["params"]

    def set_model(self):
        logging.info("Start")
        self.model = DecisionTreeRegressor(random_state=SEED)

    def fit_and_predict(self):
        logging.info("Start")
        self.fit()
        self.predict()

    def fit(self):
        logging.info("Start")
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        logging.info("Start")
        self.predicted = self.model.predict(self.X_test)


if __name__ == "__main__":
    model = ModelDecisionTreeRegressor(model_folder="decision_tree_regressor")
    model.load_data()
    model.set_model()
    model.grid_search()

    # model.load_data()
    # model.load_grid()
    # model.set_model()
    # model.fit_and_predict()
    # model.plot_metrics()
    # model.plot_wandb()
