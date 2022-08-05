import logging

from model_type import ModelType
from plot_result import PlotResults

logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-27s][%(funcName)-17s][%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename="data/log/model_runner.log"),
    ],
)


class ModelRunner:
    def __init__(self, model: list):
        logging.info("Inicializando modelos...")
        self.models = []
        if "decision_tree_regressor" in model:
            from decision_tree_regressor.decision_tree_regressor import (
                ModelDecisionTreeRegressor,
            )

            self.decision_tree_wo_features = ModelDecisionTreeRegressor(
                model_folder="decision_tree_regressor",
                model_type=ModelType.WITHOUT_FEATURES,
            )

            self.decision_tree = ModelDecisionTreeRegressor(
                model_folder="decision_tree_regressor",
                model_type=ModelType.WITH_FEATURES,
            )

            self.models.append(self.decision_tree_wo_features)
            self.models.append(self.decision_tree)
        elif "random_forest_regressor" in model:
            from random_forest_regressor.random_forest_regressor import (
                ModelRandomForestRegressor,
            )

            self.random_forest_wo_features = ModelRandomForestRegressor(
                model_folder="random_forest_regressor",
                model_type=ModelType.WITHOUT_FEATURES,
            )

            self.random_forest = ModelRandomForestRegressor(
                model_folder="random_forest_regressor",
                model_type=ModelType.WITH_FEATURES,
            )
            self.models.append(self.random_forest_wo_features)
            self.models.append(self.random_forest)
        elif "xgb_regressor" in model:
            from xgb_regressor.xgb_regressor import ModelXGBRegressor

            self.xgbr_wo_feature = ModelXGBRegressor(
                model_folder="xgb_regressor",
                model_type=ModelType.WITHOUT_FEATURES,
            )

            self.xgbr = ModelXGBRegressor(
                model_folder="xgb_regressor",
                model_type=ModelType.WITH_FEATURES,
            )
            self.models.append(self.xgbr_wo_feature)
            self.models.append(self.xgbr)
        elif "neural_network" in model:
            from neural_network.neural_network import NeuralNetwork

            self.lstsm_wo_features = NeuralNetwork(
                model_folder="neural_network",
                model_type=ModelType.WITHOUT_FEATURES,
            )

            self.lstm = NeuralNetwork(
                model_folder="neural_network",
                model_type=ModelType.WITH_FEATURES,
            )
            self.models.append(self.lstsm_wo_features)
            self.models.append(self.lstm)

    def run(self, grid_search: bool = False):
        if grid_search:
            self.execute_grid_search()
        else:
            self.train_predict()

    def train_predict(self):
        logging.info("Start")
        for model in self.models:
            model.load_data()
            model.set_model()
            model.fit_and_predict()
            model.run_metrics()
        logging.info("Finished")

    def execute_grid_search(self):
        logging.info("Start")
        for model in self.models:
            logging.info(f"GS for {model.group_name} {str(model.model_type)}")
            if model.group_name != "LSTM":
                model.load_data()
                model.set_model()
                model.grid_search()
            else:
                logging.info("Skipping LSTM..")
        logging.info("Finished")

    def show_result(self):
        logging.info("Start")
        PlotResults().show_results()


ModelRunner(model=["xgb_regressor"]).run(grid_search=True)
# ModelRunner().run(grid_search=False)
# ModelRunner().show_result()
