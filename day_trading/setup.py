#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 21:17:56 2022
@author: Juan Beleño
"""
import os
from setuptools import setup

setup(
    name='day_trading',
    version='1.0',
    packages=[
        'day_trading',
        'day_trading.data',
        'day_trading.data.inputs',
        'day_trading.data.outputs'
    ],
    install_requires=[
        'fire==0.4.0',
        'joblib==1.2.0',
        'pandas==1.3.5',
        'requests==2.28.1',
        'scikit-learn==1.0.2',
        'xgboost==1.6.2'
    ],
    entry_points={
        'console_scripts': [
            'day_trading = day_trading.__main__:main',
        ]
    }
)