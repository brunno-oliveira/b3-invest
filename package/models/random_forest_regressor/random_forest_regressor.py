import json
import logging
import os
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
        current_path = os.path.dirname(__file__)

        if self.model_type == ModelType.WITHOUT_FEATURES:
            model_type_folder = "wo_features"
        elif self.model_type == ModelType.WITH_FEATURES:
            model_type_folder = "with_features"

        best_param_path = os.path.join(
            current_path, model_type_folder, "best_params.json"
        )

        with open(best_param_path) as json_file:
            params = json.load(json_file)

        self.model = RandomForestRegressor(
            n_estimators=params["n_estimators"], random_state=SEED
        )
