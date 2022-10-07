#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:44:46 2022
@author: Juan Beleño
"""
from .files import DayTradingFiles
from datetime import datetime, timedelta
import pandas as pd
import json

class DayTradingDataset:
    def __init__(self) -> None:
        self.tickets = ['AAPL', 'NVDA', 'TSLA']
        self.ranges = ['1mo', '3mo', '6mo', '1y', '2y']
        self.granularities = ['1m', '2m', '5m', '15m', '30m']
        self.stop_time = 60

    def get_raw_data(self):
        files = DayTradingFiles()
        with open(files.aapl_filepath) as json_file:
            response = json.load(json_file)
        return response

    def prepare_dataset(self):
        print('Getting the data for AAPL')
        raw_data = self.get_raw_data()

        print('Convert the HTTP response into a pandas DataFrame')
        timestamps = raw_data['chart']['result'][0]['timestamp']
        data = raw_data['chart']['result'][0]['indicators']['quote'][0]
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
        dataset['timestamp'] = dataset['timestamp'].astype('datetime64[ns]')
        dataset['day_of_week'] = dataset['timestamp'].dt.dayofweek
        # dataset['hour'] = dataset['timestamp'].dt.hour
        print(f'There are {dataset.shape[0]} data points.')

        # Verify that the dataset is ordered by timestamp
        dataset.sort_values(by='timestamp', ascending=True, inplace=True)

        # Moving features
        print('Calculating the cummulative features.')
        column = 'close'
        periods = [5, 10, 20, 30, 50]
        for period in periods:
            dataset.loc[:, f'Return_{period}'] = dataset[column].pct_change(period)
            dataset.loc[:, f'MovingAvg_{period}'] = dataset[column].rolling(window=period).mean().values
            dataset.loc[:, f'ExpMovingAvg_{period}'] = dataset[column].ewm(span=period, adjust=False).mean().values
            dataset.loc[:, f'Volatility_{period}'] = dataset[column].diff().rolling(period).std()

        # Define target variables
        print('Defining the targets.')
        dataset['target_high'] = dataset['high'].rolling(window=8).max().shift(-8)
        dataset['target_low'] = dataset['low'].rolling(window=8).max().shift(-8)

        # Print a sample of the dataset to very it's ordered
        print(dataset.head(16))

        return dataset

    def test_train_split(self):
        dataset = self.prepare_dataset()

        # Define timestamp ranges for datasets
        week_ago = datetime.now() - timedelta(days=7)
        print(f'A week ago: {week_ago}')

        # Split the datasets
        features = [col for col in dataset.columns if col not in ['timestamp', 'target_high', 'target_low']]

        print('Defining the training data.')
        train_df = dataset[dataset['timestamp'] <= week_ago].copy()
        files = DayTradingFiles()
        train_df.to_csv(files.train_data_filepath)
        # Sample the dataset to add a little bit of randomness before training
        train_df = train_df.sample(frac=1, ignore_index=True)
        target_high_train_df = train_df['target_high'].tolist()
        target_low_train_df = train_df['target_low'].tolist()
        features_train_df = train_df[features].copy()

        print('Defining the test data.')
        test_df = dataset[dataset['timestamp'] >= week_ago].copy()
        test_df = test_df.dropna(subset=['target_high', 'target_low'])
        target_high_test_df = test_df['target_high'].tolist()
        target_low_test_df = test_df['target_low'].tolist()
        features_test_df = test_df[features].copy()

        return (
            features_train_df, target_high_train_df, target_low_train_df,
            features_test_df, target_high_test_df, target_low_test_df,
        )
