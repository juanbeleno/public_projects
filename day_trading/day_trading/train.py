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
        self.long_watchlist = self.get_watchlist('long')
        self.short_watchlist = self.get_watchlist('short')
        self.selected_tickets = self.long_watchlist.copy()
        self.selected_tickets.extend(self.short_watchlist)
        print(self.selected_tickets)

    def get_watchlist(self, watchlist_type):
        response = []
        filepath = self.files.long_watchlist
        if watchlist_type == 'short':
            filepath = self.files.short_watchlist
        with open(filepath) as json_file:
            response = json.load(json_file)
        return response

    def get_tickets(self):
        # Get about 500 ticket (S&P 500) from Finviz Map (https://finviz.com/map.ashx)
        # self.day_trading_dataset.download_ticket_candidates()
        tickets_df = pd.read_csv(self.files.ticket_candidates)
        return tickets_df['company_code'].tolist()

    def save_training_metadata(self, split_type='test_val_train'):
        training_metadata = []
        for ticket in self.tickets:
            # Load the dataset
            if split_type == 'test_val_train':
                (
                    features_train_df, target_high_train, target_low_train,
                    features_test_df, target_high_test, target_low_test,
                    _, _, _, _
                ) = self.day_trading_dataset.test_val_train_split(ticket)
            else:
                (
                    features_train_df, target_high_train, target_low_train,
                    features_test_df, target_high_test, target_low_test
                ) = self.day_trading_dataset.test_train_split(ticket)

            print('Training the models.')
            sample_size = features_test_df.shape[0]
            # Collecting metadata about the ticket
            # Pearson correlation can be calculated with at least 2 samples
            if sample_size > 1:
                high_regressor = LinearRegression()
                low_regressor = LinearRegression()

                high_regressor.fit(features_train_df, target_high_train)
                low_regressor.fit(features_train_df, target_low_train)

                low_predictions = low_regressor.predict(features_test_df)
                high_predictions = high_regressor.predict(features_test_df)
                delta_high = target_high_test - high_predictions
                delta_low = target_low_test - low_predictions
                close = features_train_df['close'].tolist()
                try:
                    metadata = {
                        'ticket': ticket,
                        'mae_low_model': mean_absolute_error(target_low_test, low_predictions),
                        'mae_high_model': mean_absolute_error(target_high_test, high_predictions),
                        'pearson_correlation_coefficient_low_model': stats.pearsonr(target_low_test, low_predictions)[0],
                        'pearson_correlation_coefficient_high_model': stats.pearsonr(target_high_test, high_predictions)[0],
                        'price': features_train_df['close'].tolist()[0],
                        'sample_size': sample_size,
                        'p_success_buy_low_sell_high': len([x for x in range(sample_size) if (delta_high[x] >= 0 and delta_low[x] >= 0)]) / sample_size,
                        'p_success_sell_high_buy_low': len([x for x in range(sample_size) if (delta_high[x] < 0 and delta_low[x] <= 0)]) / sample_size,
                        'num_interesting_long_bets': len([x for x in range(sample_size) if (delta_high[x] >= 0 and delta_low[x] >= 0 and ((high_predictions[x] - close[x]) / close[x]) > 0.01)]),
                        'num_interesting_short_bets': len([x for x in range(sample_size) if (delta_high[x] < 0 and delta_low[x] <= 0 and ((close[x] - low_predictions[x]) / close[x]) > 0.01)])
                    }
                    print(metadata)
                    training_metadata.append(metadata)
                except IndexError:
                    print(
                        f'ERROR: Problems processing the metadata of {ticket}')

        training_metadata = pd.DataFrame(training_metadata)
        training_metadata.to_csv(self.files.tickets_metadata, index=False)
        self.save_tickets(training_metadata)

    def save_tickets(self, training_metadata):
        # I'll select the top 10 tickets where the Linear Regression
        # model have shown better performance for longs and shorts.
        num_tickets = 10
        metadata = training_metadata[training_metadata['sample_size'] > 350].copy(
        )

        # STRATEGY #1 (LONG): I'll buy low and sell high.
        long_metadata = metadata[metadata['num_interesting_long_bets'] >= 15].copy(
        )
        long_metadata = long_metadata.sort_values(
            by='p_success_buy_low_sell_high', ascending=False)
        long_metadata = long_metadata.head(num_tickets)
        self.long_watchlist = long_metadata['ticket'].tolist()
        with open(self.files.long_watchlist, 'w') as f:
            json.dump(self.long_watchlist, f)

        # STRATEGY #2 (SHORT): I'll sell high and buy low.
        short_metadata = metadata[metadata['num_interesting_short_bets'] >= 15].copy(
        )
        short_metadata = short_metadata.sort_values(
            by='p_success_sell_high_buy_low', ascending=False)
        short_metadata = short_metadata.head(num_tickets)
        self.short_watchlist = short_metadata['ticket'].tolist()
        with open(self.files.short_watchlist, 'w') as f:
            json.dump(self.short_watchlist, f)

        self.selected_tickets = self.long_watchlist.copy()
        self.selected_tickets.extend(self.short_watchlist)

    def test_strategies(self):
        self.save_training_metadata()
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
            if ticket == 'HUM':
                features_test_df.to_csv(
                    self.files.features_sample, index=False)

            if ticket in self.long_watchlist:
                strategy_metadata.extend(self.strategy_manager.get_bets_for_buy_low_sell_high(
                    ticket, features_test_df))
            else:
                strategy_metadata.extend(self.strategy_manager.get_bets_for_sell_high_buy_low(
                    ticket, features_test_df))

        strategy_metadata = pd.DataFrame(strategy_metadata)
        print(strategy_metadata)
        strategy_metadata.sort_values(
            by=['index', 'p_profit'],
            ascending=[True, False],
            inplace=True)
        bets = []
        last_index = -10
        window = 8

        ticket_strike = {ticket: 0 for ticket in self.selected_tickets}
        for bet in strategy_metadata.to_dict('records'):
            if (
                (bet['index'] - last_index < window)
                or (bet['index'] == last_index)
                # Add FINRA rules for accounts with less than 25,000 USD
                or (ticket_strike[bet['ticket']] > 2)
            ):
                continue
            last_index = bet['index']
            ticket_strike[bet['ticket']] = ticket_strike[bet['ticket']] + 1
            bets.append(bet)
        bets = pd.DataFrame(bets)
        bets.to_csv(self.files.bets_metadata, index=False)
        print(f'Estimated weekly return: {bets["result"].sum()}')

    def train_models(self):
        self.save_training_metadata(split_type='test_train')
        print('Train the selected tickets to trade.')

        for ticket in self.selected_tickets:
            # Load the dataset
            (features_df, target_high,
             target_low) = self.day_trading_dataset.get_all_dataset(ticket)

            print('Training the models.')
            high_regressor = LinearRegression()
            low_regressor = LinearRegression()

            high_regressor.fit(features_df, target_high)
            low_regressor.fit(features_df, target_low)

            print(f'Saving the {ticket} models.')
            joblib.dump(low_regressor, os.path.join(
                self.files.output_directory, f'low_model_{ticket}.joblib'))
            joblib.dump(high_regressor, os.path.join(
                self.files.output_directory, f'high_model_{ticket}.joblib'))
