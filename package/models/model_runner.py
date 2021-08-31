from os import stat
from decision_tree_regressor.decision_tree_regressor import ModelDecisionTreeRegressor
from typing import List
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-10s][%(funcName)-10s][%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename="decision_tree_regressor.log"),
    ],
)


class ModelRunner:
    def __init__(self):
        self.decision_tree_regressor = ModelDecisionTreeRegressor(
            model_folder="decision_tree_regressor"
        )

    def run(self):
        models = [self.decision_tree_regressor]
        self.execute_grid_search(models)

    @staticmethod
    def execute_grid_search(models: List):
        for model in models:
            model.load_data()
            model.set_model()
            model.grid_search()


ModelRunner().run()