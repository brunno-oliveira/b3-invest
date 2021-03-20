import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class Model:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predicted = None

    def set_model(self):
        self.model = DecisionTreeRegressor(random_state=42)
        return self.model

    def fit_and_predict(self):
        self.fit()
        self.predict()
        return self.model, self.predicted

    def fit(self):
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def predict(self):
        self.predicted = self.model.predict(self.X_test)
        return self.predicted

    def plot_metrics(self):
        print(f"r2_score : {round(r2_score(self.predicted, self.y_test),4)}")
        print(
            f"mean_squared_error: {round(mean_squared_error( self.predicted, self.y_test, squared=False),4)}"
        )
        print(
            f"mean_absolute_error: {round(mean_absolute_error( self.predicted, self.y_test),4)}"
        )

        fig, ax = plt.subplots(figsize=(25, 8))
        ax.plot(predicted)
        ax.plot(np.array(y_test))
        plt.show()