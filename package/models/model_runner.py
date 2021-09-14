import logging
import os
import pickle

from decision_tree_regressor.decision_tree_regressor import ModelDecisionTreeRegressor
from model_type import ModelType
from random_forest_regressor.random_forest_regressor import ModelRandomForestRegressor

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
    def init_model(self):
        logging.info("Inicializando modelos...")
        self.decision_tree_wo_features = ModelDecisionTreeRegressor(
            model_folder="decision_tree_regressor",
            model_type=ModelType.WITHOUT_FEATURES,
        )

        self.decision_tree = ModelDecisionTreeRegressor(
            model_folder="decision_tree_regressor",
            model_type=ModelType.WITH_FEATURES,
        )

        self.random_forest_wo_features = ModelRandomForestRegressor(
            model_folder="random_forest_regressor",
            model_type=ModelType.WITHOUT_FEATURES,
        )

        self.random_forest = ModelRandomForestRegressor(
            model_folder="random_forest_regressor",
            model_type=ModelType.WITH_FEATURES,
        )

        self.models = [
            self.decision_tree_wo_features,
            self.decision_tree,
            self.random_forest_wo_features,
            self.random_forest,
        ]

    def run(self, grid_search: bool = False):
        self.init_model()
        if grid_search:
            self.execute_grid_search()
        else:
            self.train_predict()

    def train_predict(self):
        logging.info("Start")
        for model in self.models:
            model.set_model()
            model.load_data()
            model.fit_and_predict()
            model.run_metrics()
        logging.info("Finished")

    def execute_grid_search(self):
        logging.info("Start")
        for model in self.models:
            model.load_data()
            model.set_model()
            model.grid_search()
        logging.info("Finished")

    def show_result(self):
        logging.info("Start")
        self.consolidate_results()
        self.plot_results()
        logging.info("Finished")

    def consolidate_results(self):
        logging.info("Start")
        results = {}
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        result_path = os.path.join(root_path, "data", "results")
        for model in os.listdir(result_path):
            for model_type in os.listdir(os.path.join(result_path, model)):
                pickle_path = os.path.join(
                    result_path, model, model_type, "result.pickle"
                )
                with open(pickle_path, "rb") as handle:
                    result = pickle.load(handle)
                if not (model in results):
                    results[model] = {}
                results[model][model_type] = {}
                results[model][model_type].update(result)
        logging.info("Finished")

    def plot_results(self):
        logging.info("Start")


# ModelRunner().run(grid_search=True)
# ModelRunner().run(grid_search=False)
ModelRunner().show_result()
