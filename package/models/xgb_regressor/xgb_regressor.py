import logging
import random

import numpy as np
from model_base import ModelBase
from model_type import ModelType
from xgboost import XGBRegressor

SEED = 42
GROUP_NAME = "XGBRegressor"
MODEL_NAME = "2.1"

np.random.seed(SEED)
random.seed(SEED)

logger = logging.getLogger(__name__)


class ModelXGBRegressor(ModelBase):
    def __init__(self, model_folder: str, model_type: ModelType):
        super().__init__(
            group_name=GROUP_NAME,
            model_name=MODEL_NAME,
            model_folder=model_folder,
            model_type=model_type,
        )

    def set_model(self):
        logger.info("Start")
        self.model = XGBRegressor(random_state=SEED)
