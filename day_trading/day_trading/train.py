#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 19:40:02 2022
@author: Juan Beleño
"""
from .dataset import DayTradingDataset
from xgboost import XGBRegressor

class DayTradingTrainer:
    def __init__(self) -> None:
        self.day_trading_dataset = DayTradingDataset()
        self.high_regressor = XGBRegressor()
        self.low_regressor = XGBRegressor()

    def train(self):
        (
            features_train_df, target_high_train_df, target_low_train_df,
            features_test_df, target_high_test_df, target_low_test_df,
        ) = self.day_trading_dataset.test_train_split()

        self.high_regressor.fit(features_train_df, target_high_train_df)
        self.low_regressor.fit(features_train_df, target_low_train_df)