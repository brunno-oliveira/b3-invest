import os
import logging
from sklearn.tree import DecisionTreeRegressor
from model_base import ModelBase


class ModelDecisionTreeRegressor(ModelBase):
    def __init__(self):
        super().__init__()

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
    root_path = root_path = os.path.dirname(os.path.dirname(__file__))
    model = ModelDecisionTreeRegressor()
    model.set_model()
    model.load_data()
    model.fit_and_predict()
    model.plot_metrics()
