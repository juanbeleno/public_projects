#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 19:40:02 2022
@author: Juan Beleño
"""
from .dataset import DayTradingDataset
from .files import DayTradingFiles
from .strategy_manager import StrategyManager
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy import stats
import joblib
import json
import pandas as pd
import os

class DayTradingTrainer:
    def __init__(self) -> None:
        self.day_trading_dataset = DayTradingDataset()
        self.files = DayTradingFiles()
        self.strategy_manager = StrategyManager()
        self.tickets = self.get_tickets()
        self.selected_tickets = self.get_selected_tickets()

    def get_selected_tickets(self):
        response = []
        with open(self.files.selected_tickets) as json_file:
            response = json.load(json_file)
        return response

    def get_tickets(self):
        # Get about 500 ticket from Finviz Map (https://finviz.com/map.ashx)
        self.day_trading_dataset.download_ticket_candidates()
        tickets_df = pd.read_csv(self.files.ticket_candidates)
        return tickets_df['company_code'].tolist()

    def save_training_metadata(self):
        training_metadata = []
        for ticket in self.tickets:
            # Load the dataset
            (
                features_train_df, target_high_train, target_low_train,
                features_val_df, target_high_val, target_low_val,
                _, _, _, _
            ) = self.day_trading_dataset.test_val_train_split(ticket)

            print('Training the models.')
            sample_size = features_val_df.shape[0]
            # Collecting metadata about the ticket
            # Pearson correlation can be calculated with at least 2 samples
            if sample_size > 1:
                high_regressor = LinearRegression()
                low_regressor = LinearRegression()

                high_regressor.fit(features_train_df, target_high_train)
                low_regressor.fit(features_train_df, target_low_train)

                low_predictions = low_regressor.predict(features_val_df)
                high_predictions = high_regressor.predict(features_val_df)
                delta_high = target_high_val - high_predictions
                delta_low = target_low_val - low_predictions
                metadata = {
                    'ticket': ticket,
                    'mae_low_model': mean_absolute_error(target_low_val, low_predictions),
                    'mae_high_model': mean_absolute_error(target_high_val, high_predictions),
                    'pearson_correlation_coefficient_low_model': stats.pearsonr(target_low_val, low_predictions)[0],
                    'pearson_correlation_coefficient_high_model': stats.pearsonr(target_high_val, high_predictions)[0],
                    'price': features_train_df['close'].tolist()[0],
                    'sample_size': sample_size,
                    'p_success_buy_low_sell_high': len([x for x in range(sample_size) if (delta_high[x] >= 0 and delta_low[x] >= 0)]) / sample_size
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
        self.save_tickets(training_metadata)

    def save_tickets(self, training_metadata):
        # STRATEGY: I'll buy low and sell high. I'll select the top 15 tickets
        # where the Linear Regression model have shown better performance.
        num_tickets = 15
        metadata = training_metadata[training_metadata['sample_size'] > 350].copy()
        metadata = metadata.sort_values(by='p_success_buy_low_sell_high', ascending=False)
        metadata = metadata.head(num_tickets)
        self.selected_tickets = metadata['ticket'].tolist()
        with open(self.files.selected_tickets, 'w') as f:
            json.dump(self.selected_tickets, f)

    def test_strategies(self):
        training_metadata = pd.read_csv(self.files.tickets_metadata)
        self.save_tickets(training_metadata)
        strategy_metadata = []

        for ticket in self.selected_tickets:
            # Load the dataset
            (
                features_train_df, target_high_train, target_low_train,
                features_val_df, target_high_val, target_low_val,
                features_test_df, target_high_test, target_low_test, target_close_test
            ) = self.day_trading_dataset.test_val_train_split(ticket)

            # Mix training and validation data to train before testing
            features_train_df = pd.concat([features_train_df, features_val_df])
            target_high_train.extend(target_high_val)
            target_low_train.extend(target_low_val)

            print('Training the models.')
            high_regressor = LinearRegression()
            low_regressor = LinearRegression()

            high_regressor.fit(features_train_df, target_high_train)
            low_regressor.fit(features_train_df, target_low_train)

            low_predictions = low_regressor.predict(features_test_df)
            high_predictions = high_regressor.predict(features_test_df)
            features_test_df['index'] = range(features_test_df.shape[0])
            features_test_df['low_prediction'] = low_predictions
            features_test_df['high_prediction'] = high_predictions
            features_test_df['target_high'] = target_high_test
            features_test_df['target_low'] = target_low_test
            features_test_df['target_close'] = target_close_test

            strategy_metadata.extend(self.strategy_manager.get_bets_for_buy_low_sell_high(
                ticket, features_test_df))

        strategy_metadata = pd.DataFrame(strategy_metadata)
        print(strategy_metadata)
        strategy_metadata.sort_values(
            by=['index', 'risk_reward_ratio'],
            ascending=[True, False],
            inplace=True)
        bets = []
        last_index = -10
        window = 8
        for bet in strategy_metadata.to_dict('records'):
            if (
                    (bet['index'] - last_index < window)
                    or (bet['index'] == last_index)
                ):
                continue
            last_index = bet['index']
            bets.append(bet)
        bets = pd.DataFrame(bets)
        bets.to_csv(self.files.bets_metadata, index=False)
        print(f'Estimated weekly return: {bets["result"].sum()}')

