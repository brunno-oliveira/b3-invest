import os
import sys
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import random

# currentdir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.dirname(currentdir))
from model_base import ModelBase

import logging

SEED = 42
GROUP_NAME = "DecisionTreeRegressor"
MODEL_NAME = "2.1"

np.random.seed(SEED)
random.seed(SEED)

logger = logging.getLogger(__name__)


class ModelDecisionTreeRegressor(ModelBase):
    def __init__(self, model_folder: str):
        super().__init__(
            group_name=GROUP_NAME, model_name=MODEL_NAME, model_folder=model_folder
        )

    def set_model(self):
        logger.info("Start")
        self.model = DecisionTreeRegressor(random_state=SEED)

    def fit_and_predict(self):
        logger.info("Start")
        self.fit()
        self.predict()

    def fit(self):
        logger.info("Start")
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        logger.info("Start")
        self.predicted = self.model.predict(self.X_test)
