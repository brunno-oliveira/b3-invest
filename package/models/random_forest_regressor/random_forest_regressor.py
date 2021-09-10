import logging
import random

import numpy as np
from model_base import ModelBase
from model_type import ModelType
from sklearn.ensemble import RandomForestRegressor

SEED = 42
GROUP_NAME = "RandomForestRegressor"
MODEL_NAME = "2.1"

np.random.seed(SEED)
random.seed(SEED)

logger = logging.getLogger(__name__)


class ModelRandomForestRegressor(ModelBase):
    def __init__(self, model_folder: str, model_type: ModelType):
        super().__init__(
            group_name=GROUP_NAME,
            model_name=MODEL_NAME,
            model_folder=model_folder,
            model_type=model_type,
        )

    def set_model(self):
        logger.info("Start")
        self.model = RandomForestRegressor(random_state=SEED)
