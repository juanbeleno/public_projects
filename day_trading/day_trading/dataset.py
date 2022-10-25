#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:44:46 2022
@author: Juan Beleño
"""
from urllib import request
from .files import DayTradingFiles
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import json
import os
import requests

class DayTradingDataset:
    def __init__(self) -> None:
        self.range = '60d'
        self.granularity = '5m'
        self.stop_intervals = 8
        self.files = DayTradingFiles()

    def download_ticket_candidates(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'
        }
        request = requests.get(
            'https://finviz.com/maps/sec.json?rev=317',
            headers=headers
        )
        response = request.json()

        data = []
        for category_data in response['children']:
            for subcategory_data in category_data['children']:
                for company in subcategory_data['children']:
                    data.append({
                        'category': category_data['name'],
                        'subcategory': subcategory_data['name'],
                        'company_name': company['description'],
                        'company_code': company['name'],
                        'company_value': company['value']
                    })
        tickets_data = pd.DataFrame(data)
        tickets_data.to_csv(self.files.ticket_candidates, index=False)

    def get_raw_data(self, ticket):
        response = []
        print(f'Getting data for {ticket}')
        # Get the data from the past stored manually by me (Juan Beleño)
        # collecting the data on certain dates.
        ticket_directory = os.path.join(self.files.input_directory, f'{ticket}_interval_{self.granularity}_range_{self.range}')
        filepaths = next(os.walk(ticket_directory), (None, None, []))[2]  # [] if no file

        for filepath in filepaths:
            with open(os.path.join(ticket_directory, filepath)) as json_file:
                response.append(json.load(json_file))
        """
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

        # Save the data in a folder for future training
        ticket_folder = Path(ticket_directory)
        ticket_folder.mkdir(exist_ok=True)
        ticket_filepath = os.path.join(ticket_directory, f"{datetime.today().strftime('%Y-%m-%d')}.json")
        with open(ticket_filepath, 'w') as f:
            json.dump(recent_data, f)
        """

        return response

    def prepare_dataset(self, ticket):
        print('Getting the data for the tickets')
        raw_data = self.get_raw_data(ticket)

        print('Convert the HTTP response into a pandas DataFrame')
        dataset = pd.DataFrame(columns=['timestamp', 'high', 'low', 'close', 'open', 'volume'])
        for partial_data in raw_data:
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
            dataset = pd.concat([partial_dataset, dataset])

        # Remove duplicated data
        dataset.drop_duplicates(inplace=True)

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

    def test_train_split(self, ticket):
        dataset = self.prepare_dataset(ticket)

        # Define timestamp ranges for datasets
        # week_ago = datetime.now() - timedelta(days=7)
        week_ago = datetime(2022, 10, 18)
        print(f'A week ago: {week_ago}')

        # Split the datasets
        features = [col for col in dataset.columns if col not in ['timestamp', 'target_high', 'target_low']]

        print('Defining the training data.')
        train_df = dataset[dataset['timestamp'] <= week_ago].copy()
        # files = DayTradingFiles()
        # train_df.to_csv(files.train_data_filepath, index=False)
        # Sample the dataset to add a little bit of randomness before training
        train_df = train_df.sample(frac=1, ignore_index=True)
        target_high_train = train_df['target_high'].tolist()
        target_low_train = train_df['target_low'].tolist()
        features_train_df = train_df[features].copy()

        print('Defining the test data.')
        test_df = dataset[dataset['timestamp'] >= week_ago].copy()
        test_df.dropna(subset=['target_high', 'target_low'], inplace=True)
        target_high_test = test_df['target_high'].tolist()
        target_low_test = test_df['target_low'].tolist()
        features_test_df = test_df[features].copy()

        return (
            features_train_df, target_high_train, target_low_train,
            features_test_df, target_high_test, target_low_test,
        )
