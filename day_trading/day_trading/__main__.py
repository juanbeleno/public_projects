#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 21:07:19 2022
@author: Juan Beleño
"""
import fire

from .telegram_bot import TelegramBot
from .train import DayTradingTrainer
from .model_manager import ModelManager


def test_strategies():
    trainer = DayTradingTrainer()
    trainer.test_strategies()


def train_models():
    trainer = DayTradingTrainer()
    trainer.train_models()


def predict_trades():
    model_manager = ModelManager()
    print(model_manager.get_bet())


def send_message():
    telegram_bot = TelegramBot()
    telegram_bot.send_message()


def main():
    """Expose CLI functions."""
    fire.Fire({
        'test-strategies': test_strategies,
        'train-models': train_models,
        'predict-trades': predict_trades,
        'send-message': send_message
    })
