from os import stat
from decision_tree_regressor.decision_tree_regressor import ModelDecisionTreeRegressor
from random_forest_regressor.random_forest_regressor import ModelRandomForestRegressor
from typing import List
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-10s][%(funcName)-10s][%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename="data/log/model_runner.log"),
    ],
)


class ModelRunner:
    def __init__(self):
        logging.info("Inicializando modelos...")
        self.decision_tree = ModelDecisionTreeRegressor(
            model_folder="decision_tree_regressor"
        )
        self.random_forest = ModelRandomForestRegressor(
            model_folder="random_forest_regressor"
        )

    def run(self):
        models = [self.decision_tree, self.random_forest]
        self.execute_grid_search(models)

    @staticmethod
    def execute_grid_search(models: List):
        logging.info("Start")
        for model in models:
            model.load_data()
            model.set_model()
            model.grid_search()


ModelRunner().run()