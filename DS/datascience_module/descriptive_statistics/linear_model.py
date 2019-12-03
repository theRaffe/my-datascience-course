import pandas as pd
import matplotlib.pyplot as plt  # Data visualizacion libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class LinearModel:

    __init__(self, df_model, x_features, y_target):
        self.df_model = df_model
        self.x_features = x_features
        self.y_target = y_target

    def execute_regression(self, test_size=.35, random_state=1, show_scatter=False):
        m_X = self.df_model[self.x_features]
        m_y = self.df_model[self.y_target]
        X_train, X_test, y_train, y_test = train_test_split(
            m_X, m_y, test_size=test_size, random_state=random_state)
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        self.predictions = lm.predict(X_test)
        if show_scatter:
            plt.scatter(y_test, self.predictions)

        self.df_actual_predict = pd.DataFrame({'Actual': y_test, 'Predicted': self.predictions})
        self.df_coeff = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])


