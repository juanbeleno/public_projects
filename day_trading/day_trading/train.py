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
from sklearn.metrics import mean_absolute_error
from scipy import stats
import joblib
import pandas as pd
import os

class DayTradingTrainer:
    def __init__(self) -> None:
        self.day_trading_dataset = DayTradingDataset()
        self.files = DayTradingFiles()
        # self.high_regressor = XGBRegressor()
        # self.low_regressor = XGBRegressor()
        self.high_regressor = LinearRegression()
        self.low_regressor = LinearRegression()
        self.tickets = self.get_tickets()
        self.selected_tickets = []

    def get_tickets(self):
        self.day_trading_dataset.download_ticket_candidates()
        tickets_df = pd.read_csv(self.files.ticket_candidates)
        return tickets_df['company_code'].tolist()

    def train(self):
        training_metadata = []
        for ticket in self.tickets:
            # Load the dataset
            (
                features_train_df, target_high_train, target_low_train,
                features_test_df, target_high_test, target_low_test,
            ) = self.day_trading_dataset.test_train_split(ticket)

            print('Training the models.')
            sample_size = features_test_df.shape[0]
            # Collecting metadata about the ticket
            # Pearson correlation can be calculated with at least 2 samples
            if sample_size > 1:
                self.high_regressor.fit(features_train_df, target_high_train)
                self.low_regressor.fit(features_train_df, target_low_train)

                low_predictions = self.low_regressor.predict(features_test_df)
                high_predictions = self.high_regressor.predict(features_test_df)
                delta_high = target_high_test - high_predictions
                delta_low = target_low_test - low_predictions
                metadata = {
                    'ticket': ticket,
                    'mae_low_model': mean_absolute_error(target_low_test, low_predictions),
                    'mae_high_model': mean_absolute_error(target_high_test, high_predictions),
                    'pearson_correlation_coefficient_low_model': stats.pearsonr(target_low_test, low_predictions)[0],
                    'pearson_correlation_coefficient_high_model': stats.pearsonr(target_high_test, high_predictions)[0],
                    'price': features_train_df['close'].tolist()[0],
                    'sample_size': sample_size,
                    'p_high_successful_predictions': len([x for x in delta_high if x >= 0]) / sample_size,
                    'p_low_successful_predictions': len([x for x in delta_low if x <= 0]) / sample_size,
                    'p_successful_predictions': len([x for x in range(sample_size) if (delta_high[x] >= 0 and delta_low[x] <= 0)]) / sample_size
                }
                print(metadata)
                training_metadata.append(metadata)

                # print('Saving the models and the test datasets.')
                # joblib.dump(self.low_regressor, os.path.join(files.output_directory, f'low_model_{ticket}.joblib'))
                # joblib.dump(self.high_regressor, os.path.join(files.output_directory, f'high_model_{ticket}.joblib'))
                # features_test_df['target_high'] = target_high_test_df
                # features_test_df['target_low'] = target_low_test_df
                # features_test_df.to_csv(os.path.join(files.output_directory, f'validation_{ticket}.csv'), index=False)
        training_metadata = pd.DataFrame(training_metadata)
        training_metadata.to_csv(self.files.tickets_metadata, index=False)
