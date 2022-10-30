#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 21:07:19 2022
@author: Juan Beleño
"""
import fire

from .train import DayTradingTrainer

def train_models():
    trainer = DayTradingTrainer()
    trainer.test_strategies()

def main():
    """Expose CLI functions."""
    fire.Fire({
        'train-models': train_models
    })