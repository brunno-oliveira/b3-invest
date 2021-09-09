import logging
import random

import numpy as np

# currentdir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.dirname(currentdir))
from model_base import ModelBase
from model_type import ModelType
from sklearn.tree import DecisionTreeRegressor

SEED = 42
GROUP_NAME = "DecisionTreeRegressor"
MODEL_NAME = "2.1"

np.random.seed(SEED)
random.seed(SEED)

logger = logging.getLogger(__name__)


class ModelDecisionTreeRegressor(ModelBase):
    def __init__(self, model_folder: str, model_type: ModelType):
        super().__init__(
            group_name=GROUP_NAME,
            model_name=MODEL_NAME,
            model_folder=model_folder,
            model_type=model_type,
        )

    def set_model(self):
        logger.info("Start")
        self.model = DecisionTreeRegressor(random_state=SEED)
