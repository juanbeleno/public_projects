#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:39:03 2022

@author: juanbeleno
"""


class StrategyManager():
    def __init__(self) -> None:
        self.p_take_profit = 0.0075

    def get_long_metadata(self, ticket, bets):
        metadata = []
        for item in bets.to_dict('records'):
            result = (item['target_close'] - item['close']) / item['close']
            if item['label_close'] == item['close_prediction']:
                result = self.p_take_profit

            metadata.append({
                'action': item['action'],
                'ticket': ticket,
                'index': item['index'],
                'model_confidence': item['model_confidence'],
                'result': result
            })
        return metadata

    def get_short_metadata(self, ticket, bets):
        metadata = []
        for item in bets.to_dict('records'):
            result = (item['target_close'] - item['close']) / item['close']
            if item['label_close'] == item['close_prediction']:
                result = self.p_take_profit

            metadata.append({
                'action': item['action'],
                'ticket': ticket,
                'index': item['index'],
                'model_confidence': item['model_confidence'],
                'result': result
            })
        return metadata

    def get_bets_for_longs(self, ticket, df):
        # STRATEGY: I'll buy low and sell high.
        bets = df.query('close_prediction == 1').copy()
        bets['action'] = 'BUY'
        return self.get_long_metadata(ticket, bets)

    def get_bets_for_shorts(self, ticket, df):
        # STRATEGY: I'll sell high and buy low.
        bets = df.query('close_prediction == 1').copy()
        bets['action'] = 'SELL'
        return self.get_short_metadata(ticket, bets)
