#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 19:40:02 2022
@author: Juan Beleño
"""
from .dataset import DayTradingDataset
from .files import DayTradingFiles
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import joblib
import json

class DayTradingTrainer:
    def __init__(self) -> None:
        self.day_trading_dataset = DayTradingDataset()
        # self.high_regressor = XGBRegressor()
        # self.low_regressor = XGBRegressor()
        self.high_regressor = LinearRegression()
        self.low_regressor = LinearRegression()

    def train(self):
        # Load the dataset
        (
            features_train_df, target_high_train_df, target_low_train_df,
            features_test_df, target_high_test_df, target_low_test_df,
        ) = self.day_trading_dataset.test_train_split()

        print('Training the models.')
        self.high_regressor.fit(features_train_df, target_high_train_df)
        self.low_regressor.fit(features_train_df, target_low_train_df)

        print('Saving the models and the test datasets.')
        files = DayTradingFiles()
        joblib.dump(self.low_regressor, files.low_model_filepath)
        joblib.dump(self.high_regressor, files.high_model_filepath)
        features_test_df.to_csv(files.features_filepath, index=False)
        with open(files.target_high_filepath, 'w') as f:
            json.dump(target_high_test_df, f)
        with open(files.target_low_filepath, 'w') as f:
            json.dump(target_low_test_df, f)
