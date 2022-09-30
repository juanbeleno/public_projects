#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:44:46 2022
@author: Juan Beleño
"""
from datetime import datetime, timedelta
import pandas as pd
import requests

class DayTradingDataset:
    def __init__(self) -> None:
        self.tickets = ['APPL', 'NVDA', 'TSLA']
        self.ranges = ['1mo', '3mo', '6mo', '1y', '2y']
        self.granularities = ['1m', '2m', '5m', '15m', '30m']
        self.stop_time = 60

    def get_raw_data(self, ticket, range, granularity):
        yahoo_finance_url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticket}?region=US&lang=en-US&includePrePost=false&interval={granularity}&useYfid=true&range={range}&corsDomain=finance.yahoo.com&.tsrc=finance'
        r = requests.get(yahoo_finance_url)
        response = r.json()
        return response

    def prepare_dataset(self):
        raw_data = self.get_raw_data('APPL', '3mo', '5m')
        timestamps = raw_data['chart']['result']['timestamp']
        data = raw_data['chart']['result']['indicators']['quote']
        dataset = pd.DataFrame({
            'timestamp': timestamps,
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'open': data['open'],
            'volume': data['volume']
        })

        # Convert int to timestamp
        dataset['timestamp'] = dataset['timestamp'].apply(
            lambda timestamp: pd.to_datetime(timestamp, utc=True, unit='s')
        )

        # Moving features
        column = 'close'
        periods = [5, 10, 20, 30, 50]
        for period in periods:
            dataset.loc[:,'Return_{period}'] = dataset[column].pct_change(period)
            dataset.loc[:,'MovingAvg_{period}'] = dataset[column].rolling(window=period).mean().values
            dataset.loc[:,'ExpMovingAvg_{period}'] = dataset[column].ewm(span=period, adjust=False).mean().values
            dataset.loc[:,'Volatility_{period}'] = dataset[column].diff().rolling(period).std()

        # Define target variables
        dataset['target_high'] = dataset['high'].rolling(window=8).max().shift(-8)
        dataset['target_low'] = dataset['low'].rolling(window=8).max().shift(-8)
        return dataset

    def test_train_split(self):
        dataset = self.prepare_dataset()

        # Define timestamp ranges for datasets
        week_ago = datetime.now() - timedelta(days=7)

        # Split the datasets
        features = [col for col in dataset.columns if col not in ['timestamp', 'target_high', 'target_low']]

        train_df = dataset[dataset['timestamp'] <= week_ago].copy()
        train_df.sample(frac=1, ignore_index=True, inplace=True)
        target_high_train_df = train_df['target_high'].tolist()
        target_low_train_df = train_df['target_low'].tolist()
        features_train_df = train_df[features].copy()

        test_df = dataset[dataset['timestamp'] >= week_ago].copy()
        test_df = test_df.dropna(subset=['target_high', 'target_low'])
        target_high_test_df = test_df['target_high'].tolist()
        target_low_test_df = test_df['target_low'].tolist()
        features_test_df = test_df[features].copy()

        return (
            features_train_df, target_high_train_df, target_low_train_df,
            features_test_df, target_high_test_df, target_low_test_df,
        )
