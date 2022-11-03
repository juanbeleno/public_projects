#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:39:03 2022

@author: juanbeleno
"""


class StrategyManager():
    def get_bet_metadata(self, ticket, bets):
        metadata = []
        for item in bets.to_dict('records'):
            result = (item['target_close'] - item['close']) / item['close']
            if item['target_low'] < item['low_prediction']:
                result = (item['low_prediction'] -
                          item['close']) / item['close']
            elif item['target_high'] > item['high_prediction']:
                result = (item['high_prediction'] -
                          item['close']) / item['close']

            metadata.append({
                'ticket': ticket,
                'index': item['index'],
                'close': item['close'],
                'take_profit': item['high_prediction'],
                'stop_loss': item['low_prediction'],
                'risk_reward_ratio': (item['high_prediction'] - item['close']) / (item['close'] - item['low_prediction']),
                'result': result
            })
        return metadata

    def get_bets_for_buy_low_sell_high(self, ticket, df):
        # STRATEGY: I'll buy low and sell high. Risk/Reward ratio 1:3.
        # At least a variation of 1% of the ticket value.
        bets = df.query(
            'high_prediction > close and ((high_prediction - close) > 3 * (close - low_prediction)) and ((high_prediction - close) / close) > 0.01')
        return self.get_bet_metadata(ticket, bets)
