import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from model_base import ModelBase
import logging

SEED = 42
GROUP_NAME = "RandomForestRegressor"
MODEL_NAME = "2.1"

np.random.seed(SEED)
random.seed(SEED)

logger = logging.getLogger(__name__)


class ModelRandomForestRegressor(ModelBase):
    def __init__(self, model_folder: str):
        super().__init__(
            group_name=GROUP_NAME, model_name=MODEL_NAME, model_folder=model_folder
        )

    def set_model(self):
        logger.info("Start")
        self.model = RandomForestRegressor(random_state=SEED)