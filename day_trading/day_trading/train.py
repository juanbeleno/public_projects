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
import os

class DayTradingTrainer:
    def __init__(self) -> None:
        self.day_trading_dataset = DayTradingDataset()
        # self.high_regressor = XGBRegressor()
        # self.low_regressor = XGBRegressor()
        self.high_regressor = LinearRegression()
        self.low_regressor = LinearRegression()
        self.tickets = [
            'AAPL',
            'NVDA',
            'TSLA',
            'GOOG',
            'META',
            'MSFT',
            'QQQ',
            'SPY'
        ]

    def train(self):
        for ticket in self.tickets:
            # Load the dataset
            (
                features_train_df, target_high_train_df, target_low_train_df,
                features_test_df, target_high_test_df, target_low_test_df,
            ) = self.day_trading_dataset.test_train_split(ticket)

            print('Training the models.')
            self.high_regressor.fit(features_train_df, target_high_train_df)
            self.low_regressor.fit(features_train_df, target_low_train_df)

            print('Saving the models and the test datasets.')
            files = DayTradingFiles()
            joblib.dump(self.low_regressor, os.path.join(files.output_directory, f'low_model_{ticket}.joblib'))
            joblib.dump(self.high_regressor, os.path.join(files.output_directory, f'high_model_{ticket}.joblib'))
            features_test_df.to_csv(os.path.join(files.output_directory, f'features_{ticket}.csv'), index=False)
            with open(os.path.join(files.output_directory, f'target_high_{ticket}.json'), 'w') as f:
                json.dump(target_high_test_df, f)
            with open(os.path.join(files.output_directory, f'target_low_{ticket}.json'), 'w') as f:
                json.dump(target_low_test_df, f)
