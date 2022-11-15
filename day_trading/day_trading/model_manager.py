
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 21:37:44 2022
@author: Juan Beleño
"""
from .files import DayTradingFiles
from .dataset import DayTradingDataset
import json
import joblib
import os
import pandas as pd


class ModelManager:
    def __init__(self) -> None:
        self.files = DayTradingFiles()
        self.day_trading_dataset = DayTradingDataset()
        self.long_watchlist = self.get_watchlist('long')
        self.short_watchlist = self.get_watchlist('short')
        self.selected_tickets = self.long_watchlist.copy()
        self.selected_tickets.extend(self.short_watchlist)
        self.high_models = {
            ticket: self.get_model(ticket, 'high')
            for ticket in self.selected_tickets
        }
        self.low_models = {
            ticket: self.get_model(ticket, 'low')
            for ticket in self.selected_tickets
        }
        self.default_bet = {
            'ticket': 'NONE',
            'stop_loss': 0,
            'take_profit': 0,
            'p_profit': 0
        }
        self.profit_ratio = 1.75

    def get_watchlist(self, watchlist_type):
        response = []
        filepath = self.files.long_watchlist
        if watchlist_type == 'short':
            filepath = self.files.short_watchlist
        with open(filepath) as json_file:
            response = json.load(json_file)
        return response

    def get_model(self, ticket, model_type):
        model = joblib.load(os.path.join(
            self.files.output_directory, f'{model_type}_model_{ticket}.joblib'))
        return model

    def get_bet(self):
        possible_bets = []
        money = int(os.environ['MONEY'])

        for ticket in self.selected_tickets:
            low_model = self.low_models[ticket]
            high_model = self.high_models[ticket]

            features = self.day_trading_dataset.get_prediction_features(ticket)
            try:
                # By default, I'll calculate the variables for LONG
                action = 'BUY'
                low_prediction = low_model.predict(features)[0]
                high_prediction = high_model.predict(features)[0]
                close = features['close'].tolist()[0]
                p_profit = (high_prediction - close) / close
                p_loss = (close - low_prediction) / close
                stop_loss = (close - low_prediction) * money / close
                take_profit = (high_prediction - close) * money / close

                # SHORT
                if ticket in self.short_watchlist:
                    action = 'SELL'
                    p_profit = (close - low_prediction) / close
                    p_loss = (high_prediction - close) / close
                    stop_loss = (high_prediction - close) * money / close
                    take_profit = (close - low_prediction) * money / close

                possible_bets.append({
                    'action': action,
                    'ticket': ticket,
                    'close': close,
                    'p_profit': p_profit,
                    'p_loss': p_loss,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit})
            except ValueError:
                print(
                    f'ERROR: Problems making predictions using recent data for {ticket}')
        possible_bets = pd.DataFrame(possible_bets)
        possible_bets = possible_bets.query(
            f'p_profit > {self.profit_ratio} * p_loss').copy()
        possible_bets.sort_values(
            by='p_profit',
            ascending=False,
            inplace=True)
        bet = self.default_bet
        if possible_bets.shape[0] > 0:
            bet = possible_bets.to_dict('records')[0]
        return bet
