#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 19:40:02 2022
@author: Juan Beleño
"""
from .dataset import DayTradingDataset
from .files import DayTradingFiles
from .strategy_manager import StrategyManager
# from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib
import json
import pandas as pd
import os


class DayTradingTrainer:
    def __init__(self) -> None:
        self.day_trading_dataset = DayTradingDataset()
        self.window = self.day_trading_dataset.window
        self.files = DayTradingFiles()
        self.strategy_manager = StrategyManager()
        self.tickets = self.get_tickets()

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

    def save_training_metadata(self, split_type='test_val_train', bet_type='long'):
        training_metadata = []
        for ticket in self.tickets:
            # Load the dataset
            if split_type == 'test_val_train':
                (
                    features_train_df, label_close_train,
                    features_test_df, label_close_test,
                    _, _, _
                ) = self.day_trading_dataset.test_val_train_split(ticket, bet_type)
            else:
                (
                    features_train_df, label_close_train,
                    features_test_df, label_close_test
                ) = self.day_trading_dataset.test_train_split(ticket, bet_type)

            sample_size = features_test_df.shape[0]
            num_bets_threshold = 30 * 2.5
            num_bets = sum(label_close_test)
            if num_bets > num_bets_threshold and sample_size:
                # Collecting metadata about the ticket
                print('Training the models.')
                close_model = CatBoostClassifier()
                close_model.fit(features_train_df,
                                label_close_train, verbose=False)
                close_predictions = close_model.predict(
                    features_test_df, verbose=False)
                num_interesting_bets = sum(close_predictions)
                metadata = {
                    'ticket': ticket,
                    'accuracy': accuracy_score(label_close_test, close_predictions),
                    'precision': precision_score(label_close_test, close_predictions),
                    'recall': recall_score(label_close_test, close_predictions),
                    'f1_score': f1_score(label_close_test, close_predictions),
                    'sample_size': sample_size,
                    'num_interesting_bets': num_interesting_bets,
                    'num_possible_bets': num_bets
                }
                print(metadata)
                training_metadata.append(metadata)
            else:
                print('Not enough interesting bets to train a model on this ticket')

        training_metadata = pd.DataFrame(training_metadata)
        training_metadata_filepath = self.files.long_tickets_metadata
        if bet_type == 'short':
            training_metadata_filepath = self.files.short_tickets_metadata
        training_metadata.to_csv(training_metadata_filepath, index=False)
        self.save_tickets(training_metadata, bet_type)

    def save_tickets(self, training_metadata, bet_type):
        # I'll select the top 3 tickets where the Logistic Regression
        # model have shown better performance for the bet_type.
        num_tickets = 3
        num_bets_threshold = 30
        metadata = training_metadata[training_metadata['sample_size'] > 700].copy(
        )
        metadata = metadata[metadata['num_interesting_bets'] >= num_bets_threshold].copy(
        )
        metadata = metadata.query(
            f'precision > 0.66').copy()
        metadata = metadata.sort_values(
            by='precision', ascending=False)
        metadata = metadata.head(num_tickets)

        watchlist = metadata['ticket'].tolist()
        watchlist_filepath = self.files.long_watchlist
        if bet_type == 'short':
            watchlist_filepath = self.files.short_watchlist
        with open(watchlist_filepath, 'w') as f:
            json.dump(watchlist, f)

    def test_strategies(self):
        strategy_metadata = []
        for bet_type in ['long', 'short']:
            self.save_training_metadata(bet_type=bet_type)

            for ticket in self.get_watchlist(bet_type):
                # Load the dataset
                (
                    features_train_df, label_close_train,
                    features_val_df, label_close_val,
                    features_test_df, label_close_test, target_close_test
                ) = self.day_trading_dataset.test_val_train_split(ticket)

                # Mix training and validation data to train before testing
                features_train_df = pd.concat(
                    [features_train_df, features_val_df])
                label_close_train.extend(label_close_val)

                print('Training the models.')
                close_model = CatBoostClassifier()
                close_model.fit(features_train_df,
                                label_close_train, verbose=False)
                close_predictions = close_model.predict(features_test_df)
                model_confidence = close_model.predict_proba(features_test_df)[
                    :, 1]
                features_test_df['index'] = range(features_test_df.shape[0])
                features_test_df['close_prediction'] = close_predictions
                features_test_df['model_confidence'] = model_confidence
                features_test_df['label_close'] = label_close_test
                features_test_df['target_close'] = target_close_test

                if bet_type == 'long':
                    strategy_metadata.extend(self.strategy_manager.get_bets_for_longs(
                        ticket, features_test_df))
                else:
                    strategy_metadata.extend(self.strategy_manager.get_bets_for_shorts(
                        ticket, features_test_df))

        strategy_metadata = pd.DataFrame(strategy_metadata)
        print(strategy_metadata)
        strategy_metadata.sort_values(
            by=['index', 'model_confidence'],
            ascending=[True, False],
            inplace=True)
        bets = []
        last_index = -10

        ticket_strike = {
            ticket: 0 for ticket in strategy_metadata['ticket'].unique()}
        for bet in strategy_metadata.to_dict('records'):
            if (
                (bet['index'] - last_index < self.window)
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
        for bet_type in ['long', 'short']:
            self.save_training_metadata(
                split_type='test_train', bet_type=bet_type)
            print('Train the selected tickets to trade.')

            for ticket in self.get_watchlist(bet_type):
                # Load the dataset
                (features_df, label_close) = self.day_trading_dataset.get_all_dataset(
                    ticket, bet_type)

                print('Training the models.')
                close_model = CatBoostClassifier()
                close_model.fit(features_df, label_close, verbose=False)

                print(f'Saving the {ticket} models.')
                joblib.dump(close_model, os.path.join(
                    self.files.output_directory, f'close_model_{ticket}.joblib'))
