import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Data visualizacion libraries
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class RegressionModel:

    def __init__(self, df_model, x_features, y_target):
        self.df_model = df_model
        self.x_features = x_features
        self.y_target = y_target

    def linear_model(self, df_source, test_percent=3.5):
        sns.pairplot(df_source)

    def execute_regression(self, test_size=3.5, random_state=1):
        m_X = self.df_model[x_features]
        m_y = self.df_model[y_target]
        X_train, X_test, y_train, y_test = train_test_split(
            m_X, m_y, test_size=test_size, random_state=random_state)
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        self.predictions = lm.predict(X_test)
        plt.scatter(y_test, self.predictions)
