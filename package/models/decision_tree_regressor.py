from sklearn.tree import DecisionTreeRegressor
from model_base import ModelBase
import logging

GROUP_NAME = "DecisionTreeRegressor"
MODEL_NAME = "1.0"

# import os
# os.environ["WANDB_MODE"] = "dryrun"


class ModelDecisionTreeRegressor(ModelBase):
    def __init__(self):
        super().__init__(group_name=GROUP_NAME, model_name=MODEL_NAME)

    def set_model(self):
        logging.info("Start")
        self.model = DecisionTreeRegressor(random_state=42)
        return self.model

    def fit_and_predict(self):
        logging.info("Start")
        self.fit()
        self.predict()
        return self.model, self.predicted

    def fit(self):
        logging.info("Start")
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def predict(self):
        logging.info("Start")
        self.predicted = self.model.predict(self.X_test)
        return self.predicted


if __name__ == "__main__":
    model = ModelDecisionTreeRegressor()
    model.set_model()
    model.load_data()
    model.fit_and_predict()
    model.plot_metrics()
    model.plot_wandb()
