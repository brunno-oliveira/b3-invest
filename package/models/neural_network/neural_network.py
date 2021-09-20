import logging
import random

import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from model_base import ModelBase
from model_type import ModelType

import tensorflow as tf

SEED = 42
GROUP_NAME = "LSTM"
MODEL_NAME = "2.1"

np.random.seed(SEED)
random.seed(SEED)

logger = logging.getLogger(__name__)


class NeuralNetwork(ModelBase):
    def __init__(self, model_folder: str, model_type: ModelType):
        super().__init__(
            group_name=GROUP_NAME,
            model_name=MODEL_NAME,
            model_folder=model_folder,
            model_type=model_type,
        )

    def set_model(self):
        logger.info("Start")
        self.model = Sequential()

        self.model.add(
            LSTM(
                units=50,
                return_sequences=True,
                input_shape=(self.X_train.shape[0], self.X_train.shape[1]),
            )
        )
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(units=1))

        self.model.compile(optimizer="adam", loss="mean_squared_error")
        self.model.summary()

    def fit_and_predict(self):
        logger.info("Start")
        self.reshape()
        self.fit()
        self.predict()

    def fit(self):
        logger.info("Start")

        self.model.fit(self.X_train, self.y_train, epochs=100, verbose=2)
        logger.info("Done")

    def predict(self):
        logger.info("Predict 1 day..")
        self.predicted_1_day = self.model.predict(self.X_test_1_day)
        self.test_data_1_day["predicted"] = self.predicted_1_day
        self.model_result["1_day"].update({"data": self.test_data_1_day.to_dict()})

        logger.info("Predict 7 days..")
        self.predicted_7_days = self.model.predict(self.X_test_7_days)
        self.test_data_7_days["predicted"] = self.predicted_7_days
        self.model_result["7_days"].update({"data": self.test_data_7_days.to_dict()})

        logger.info("Predict 14 days..")
        self.predicted_14_days = self.model.predict(self.X_test_14_days)
        self.test_data_14_days["predicted"] = self.predicted_14_days
        self.model_result["14_days"].update({"data": self.test_data_14_days.to_dict()})

        logger.info("Predict 28 days..")
        self.predicted_28_days = self.model.predict(self.X_test_28_days)
        self.test_data_28_days["predicted"] = self.predicted_28_days
        self.model_result["28_days"].update({"data": self.test_data_28_days.to_dict()})

        logger.info("Done")

    def reshape(self):
        """O Keras não aceita um Dataframe como input"""
        # self.X_train = self.X_train[:200]
        # self.y_train = self.y_train[:200]

        # Train
        self.X_train = self.reshape_x(self.X_train)
        self.y_train = self.reshape_y(self.y_train)

        # Test
        self.X_test_1_day = self.reshape_x(self.X_test_1_day)
        self.y_test_1_day = self.reshape_y(self.y_test_1_day)

        self.X_test_7_days = self.reshape_x(self.X_test_7_days)
        self.y_test_7_days = self.reshape_y(self.y_test_7_days)

        self.X_test_14_days = self.reshape_x(self.X_test_14_days)
        self.y_test_14_days = self.reshape_y(self.y_test_14_days)

        self.X_test_28_days = self.reshape_x(self.X_test_28_days)
        self.y_test_28_days = self.reshape_y(self.y_test_28_days)

    @staticmethod
    def reshape_x(x):
        """Retorna o X no formato que o Keras espera"""
        return np.array(x).reshape(x.shape[0], 1, x.shape[1])

    @staticmethod
    def reshape_y(y):
        """Retorna o y no formato que o Keras espera"""
        return np.array(y).reshape(y.shape[0], 1)
