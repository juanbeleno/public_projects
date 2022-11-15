#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:39:03 2022

@author: juanbeleno
"""


class StrategyManager():
    def __init__(self) -> None:
        # 0.32% is the minimum stop loss allowed by eToro
        # And we are using a 1:1.75 Risk-Reward Ratio
        self.profit_ratio = 1.625
        self.p_profit_threshold = 0.0033 * self.profit_ratio

    def get_long_metadata(self, ticket, bets):
        metadata = []
        for item in bets.to_dict('records'):
            bottom_limit = item['close'] * \
                (1 - self.p_profit_threshold/self.profit_ratio)
            if item['low_prediction'] < bottom_limit:
                bottom_limit = item['low_prediction']
            result = (item['target_close'] - item['close']) / item['close']
            if item['target_low'] <= bottom_limit:
                result = -(self.p_profit_threshold / self.profit_ratio)
            elif item['target_high'] >= item['close'] * (1 + self.p_profit_threshold):
                result = self.p_profit_threshold

            metadata.append({
                'action': item['action'],
                'ticket': ticket,
                'index': item['index'],
                'close': item['close'],
                'p_profit': (item['high_prediction'] - item['close']) / item['close'],
                'take_profit': item['high_prediction'],
                'stop_loss': item['low_prediction'],
                'risk_reward_ratio': (item['high_prediction'] - item['close']) / (item['close'] - item['low_prediction']),
                'result': result
            })
        return metadata

    def get_short_metadata(self, ticket, bets):
        metadata = []
        for item in bets.to_dict('records'):
            result = (item['target_close'] - item['close']) / item['close']
            top_limit = item['close'] * \
                (1 + self.p_profit_threshold/self.profit_ratio)
            if item['high_prediction'] > top_limit:
                top_limit = item['high_prediction']
            if item['target_high'] >= top_limit:
                result = -(self.p_profit_threshold / self.profit_ratio)
            elif item['target_low'] <= item['close'] * (1 - self.p_profit_threshold):
                result = self.p_profit_threshold

            metadata.append({
                'action': item['action'],
                'ticket': ticket,
                'index': item['index'],
                'close': item['close'],
                'p_profit': (item['close'] - item['low_prediction']) / item['close'],
                'take_profit': item['low_prediction'],
                'stop_loss': item['high_prediction'],
                'risk_reward_ratio': (item['close'] - item['low_prediction']) / (item['high_prediction'] - item['close']),
                'result': result
            })
        return metadata

    def get_bets_for_buy_low_sell_high(self, ticket, df):
        # STRATEGY: I'll buy low and sell high. Risk/Reward ratio 2:3.
        # At least a variation of 1% of the ticket value.
        bets = df.query(
            f'high_prediction > close and ((high_prediction - close) > {self.profit_ratio} * (close - low_prediction)) and ((high_prediction - close) / close) > {self.p_profit_threshold}').copy()
        bets['action'] = 'BUY'
        return self.get_long_metadata(ticket, bets)

    def get_bets_for_sell_high_buy_low(self, ticket, df):
        # STRATEGY: I'll sell high and buy low. Risk/Reward ratio 2:3.
        # At least a variation of 1% of the ticket value.
        bets = df.query(
            f'low_prediction < close and ((close - low_prediction) > {self.profit_ratio} * (high_prediction - close)) and ((close - low_prediction) / close) > {self.p_profit_threshold}').copy()
        bets['action'] = 'SELL'
        return self.get_short_metadata(ticket, bets)
