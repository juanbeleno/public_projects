
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
        self.tickets = self.get_selected_tickets()
        self.high_models = {
            ticket: self.get_model(ticket, 'high')
            for ticket in self.get_selected_tickets()
        }
        self.low_models = {
            ticket: self.get_model(ticket, 'low')
            for ticket in self.get_selected_tickets()
        }

    def get_selected_tickets(self):
        response = []
        with open(self.files.selected_tickets) as json_file:
            response = json.load(json_file)
        return response

    def get_model(self, ticket, model_type):
        model = joblib.load(os.path.join(
            self.files.output_directory, f'{model_type}_model_{ticket}.joblib'))
        return model

    def get_bet(self):
        possible_bets = []
        for ticket in self.tickets:
            low_model = self.low_models[ticket]
            high_model = self.high_models[ticket]

            features = self.day_trading_dataset.get_prediction_features(ticket)
            low_prediction = low_model.predict(features)[0]
            high_prediction = high_model.predict(features)[0]
            close = features['close'].tolist()[0]
            possible_bets.append({
                'ticket': ticket,
                'close': close,
                'market_order_bottom': low_prediction,
                'market_order_top': high_prediction,
                'min_market_order_bottom': close - (high_prediction - close) * 0.2,
                'p_earnings': ((high_prediction - close) / close),
                'risk_reward_ratio': (high_prediction - close) / (close - low_prediction)
            })
        possible_bets = pd.DataFrame(possible_bets)
        possible_bets.sort_values(
            by='risk_reward_ratio',
            ascending=False,
            inplace=True)
        return possible_bets.to_dict('records')[0]
