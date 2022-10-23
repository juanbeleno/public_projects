#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:44:46 2022
@author: Juan Beleño
"""
from urllib import request
from .files import DayTradingFiles
from datetime import datetime, timedelta
import pandas as pd
import json
import os
import requests

class DayTradingDataset:
    def __init__(self) -> None:
        self.tickets = [
            'AAPL',
            'NVDA',
            'TSLA',
            'GOOG',
            'META',
            'MSFT',
            'QQQ',
            'SPY'
        ]
        self.range = '60d'
        self.granularity = '5m'
        self.stop_intervals = 8

    def get_raw_data(self):
        response = []
        for ticket in self.tickets:
            print(f'Getting data for {ticket}')
            # Get the data from the past stored manually by me (Juan Beleño)
            # collecting the data on certain dates.
            files = DayTradingFiles()
            ticket_directory = os.path.join(files.input_directory, f'{ticket}_interval_{self.granularity}_range_{self.range}')
            filepaths = next(os.walk(ticket_directory), (None, None, []))[2]  # [] if no file

            for filepath in filepaths:
                with open(os.path.join(ticket_directory, filepath)) as json_file:
                    response.append(json.load(json_file))

            # Collect the most recent data using the Yahoo Finances API
            params = {
                'region': 'US',
                'lang': 'en-US',
                'includePrePost': 'false',
                'interval': self.granularity,
                'useYfid': 'true',
                'range': self.range,
                'corsDomain': 'finance.yahoo.com',
                '.tsrc': 'finance'
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'
            }
            request = requests.get(
                f'https://query1.finance.yahoo.com/v8/finance/chart/{ticket}',
                params=params,
                headers=headers
            )
            recent_data = request.json()
            response.append(recent_data)

        return response

    def prepare_dataset(self):
        print('Getting the data for the tickets')
        raw_data = self.get_raw_data()

        print('Convert the HTTP response into a pandas DataFrame')
        dataset = pd.DataFrame(columns=['timestamp', 'high', 'low', 'close', 'open', 'volume', 'ticket'])
        for partial_data in raw_data:
            ticket = partial_data['chart']['result'][0]['meta']['symbol']
            print(f'Processing data for {ticket}')
            timestamps = partial_data['chart']['result'][0]['timestamp']
            data = partial_data['chart']['result'][0]['indicators']['quote'][0]
            partial_dataset = pd.DataFrame({
                'timestamp': timestamps,
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'open': data['open'],
                'volume': data['volume']
            })
            partial_dataset['ticket'] = ticket
            dataset = pd.concat([partial_dataset, dataset])

        # Remove duplicated data
        dataset.drop_duplicates(inplace=True)

        # Use only one ticket
        dataset = dataset[dataset['ticket'] == 'AAPL'].copy()

        # Create binary column with the values of each ticket
        dataset = pd.get_dummies(dataset, columns=['ticket'])

        # Convert int to timestamp
        dataset['timestamp'] = dataset['timestamp'].apply(
            lambda timestamp: pd.to_datetime(timestamp, utc=True, unit='s')
        )
        dataset['timestamp'] = dataset['timestamp'].astype('datetime64[ns]')
        dataset['day_of_week'] = dataset['timestamp'].dt.dayofweek
        dataset['hour'] = dataset['timestamp'].dt.hour
        dataset['minute'] = dataset['timestamp'].dt.minute
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

        # Remove the data points at the end of the day because
        # they have too much volatility
        # dataset = dataset[(dataset['hour'] != 19)].copy()

        # Drop NA for Linear Regression
        dataset.dropna(inplace=True)

        # Print a sample of the dataset to very it's ordered
        print(dataset.head(16))

        return dataset

    def test_train_split(self):
        dataset = self.prepare_dataset()

        # Define timestamp ranges for datasets
        week_ago = datetime.now() - timedelta(days=7)
        #week_ago = datetime(2022, 9, 30)
        print(f'A week ago: {week_ago}')

        # Split the datasets
        features = [col for col in dataset.columns if col not in ['timestamp', 'target_high', 'target_low']]

        print('Defining the training data.')
        train_df = dataset[dataset['timestamp'] <= week_ago].copy()
        files = DayTradingFiles()
        train_df.to_csv(files.train_data_filepath, index=False)
        # Sample the dataset to add a little bit of randomness before training
        train_df = train_df.sample(frac=1, ignore_index=True)
        target_high_train_df = train_df['target_high'].tolist()
        target_low_train_df = train_df['target_low'].tolist()
        features_train_df = train_df[features].copy()

        print('Defining the test data.')
        test_df = dataset[((dataset['timestamp'] >= week_ago) & (dataset['ticket_AAPL'] == 1))].copy()
        test_df.dropna(subset=['target_high', 'target_low'], inplace=True)
        target_high_test_df = test_df['target_high'].tolist()
        target_low_test_df = test_df['target_low'].tolist()
        features_test_df = test_df[features].copy()

        return (
            features_train_df, target_high_train_df, target_low_train_df,
            features_test_df, target_high_test_df, target_low_test_df,
        )
