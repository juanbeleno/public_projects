#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 11:26:02 2022
@author: Juan Beleño
"""
from .model_manager import ModelManager
import requests
import os


class TelegramBot:
    def __init__(self) -> None:
        telegram_token = os.environ['TELEGRAM_TOKEN']
        self.url = f'https://api.telegram.org/bot{telegram_token}/sendMessage'
        self.model_manager = ModelManager()

    def send_message(self):
        bet = self.model_manager.get_bet()
        # TODO: Use only bets with an increase of 0.75% in the ticket price
        if bet['p_earnings'] > 0.0075:
            message = f"Ticket: {bet['ticket']}\nLow: {bet['min_market_order_bottom']}\nHigh: {bet['market_order_top']}\nClose: {bet['close']}\nFalse Low: {bet['market_order_bottom']}"
            payload = {
                'chat_id': os.environ['CHAT_ID'],
                'text': message
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'
            }
            response = requests.post(self.url, json=payload, headers=headers)
            print(response.text)
