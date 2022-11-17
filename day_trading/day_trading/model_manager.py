
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
            'take_profit': 0
        }

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
            low_prediction = low_model.predict(features)[0]
            high_prediction = high_model.predict(features)[0]
            low_model_confidence = low_model.predict_proba(features)[:, 1]

            # By default, I'll calculate the variables for LONG
            action = 'BUY'
            stop_loss = money * self.day_trading_dataset.p_stop_loss
            take_profit = money * self.day_trading_dataset.p_take_profit

            # SHORT
            if ticket in self.short_watchlist:
                action = 'SELL'

            if low_prediction == 1 and high_prediction == 1:
                possible_bets.append({
                    'action': action,
                    'ticket': ticket,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'close': features['close'].tolist()[0],
                    'low_model_confidence': low_model_confidence})
        possible_bets = pd.DataFrame(possible_bets)
        possible_bets.sort_values(
            by='low_model_confidence',
            ascending=False,
            inplace=True)
        bet = self.default_bet
        if possible_bets.shape[0] > 0:
            bet = possible_bets.to_dict('records')[0]
        return bet
