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
import numpy as np
import pandas as pd
import json
import os
import requests
# Division by zero
np.seterr(divide='ignore', invalid='ignore')


class DayTradingDataset:
    def __init__(self) -> None:
        self.range = '60d'
        self.granularity = '5m'
        self.window = 4
        self.files = DayTradingFiles()

    def download_ticket_candidates(self):
        # S&P 500
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

    def get_recent_data(self, ticket):
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
        return recent_data

    def get_raw_data(self, ticket):
        response = []
        print(f'Getting data for {ticket}')
        # Get the data from the past stored manually by me (Juan Beleño)
        # collecting the data on certain dates.
        ticket_directory = os.path.join(
            self.files.input_directory, f'{ticket}_interval_{self.granularity}_range_{self.range}')
        filepaths = next(os.walk(ticket_directory), (None, None, []))[
            2]  # [] if no file

        for filepath in filepaths:
            with open(os.path.join(ticket_directory, filepath)) as json_file:
                response.append(json.load(json_file))

        recent_data = self.get_recent_data(ticket)
        response.append(recent_data)
        """
        # Save the data in a folder for future training
        ticket_folder = Path(ticket_directory)
        ticket_folder.mkdir(exist_ok=True)
        ticket_filepath = os.path.join(
            ticket_directory, f"{datetime.today().strftime('%Y-%m-%d')}.json")
        with open(ticket_filepath, 'w') as f:
            json.dump(recent_data, f)
        """

        return response

    def transform_data(self, ticket, raw_data):
        print('Convert the HTTP response into a pandas DataFrame')
        dataset = pd.DataFrame(
            columns=['timestamp', 'high', 'low', 'close', 'open', 'volume'])
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
            lambda timestamp: pd.to_datetime(timestamp, utc=False, unit='s')
        )
        dataset['day_of_week'] = dataset['timestamp'].dt.dayofweek
        dataset['hour'] = dataset['timestamp'].dt.hour
        dataset['minute'] = dataset['timestamp'].dt.minute
        dataset['date'] = dataset['timestamp'].dt
        print(f'There are {dataset.shape[0]} data points.')

        # Verify that the dataset is ordered by timestamp
        dataset.sort_values(by='timestamp', ascending=True,
                            inplace=True, ignore_index=True)

        # Moving features
        print('Calculating the cummulative features.')
        column = 'close'
        time_windows = []
        for window in time_windows:
            dataset.loc[:, f'Return_{window}'] = dataset[column].pct_change(
                window)
            dataset.loc[:, f'MovingAvg_{window}'] = dataset[column].rolling(
                window=window).mean().values
            dataset.loc[:, f'EMA_{window}'] = dataset[column].ewm(
                span=window, adjust=False).mean().values
            dataset.loc[:, f'Volatility_{window}'] = dataset[column].diff().rolling(
                window).std()

        # Support/Resistance (Only vertical)
        time_windows = [3, 13, 21, 34, 55]
        for window in time_windows:
            dataset.loc[:, f'Support_{window}'] = dataset['low'].rolling(
                window=window).min().values
            dataset.loc[:, f'Resistance_{window}'] = dataset['high'].rolling(
                window=window).max().values

        # RSI
        rsi_period = 14
        delta = dataset['close'] - dataset['open']
        gains, loses = delta.copy(), delta.copy()
        gains[gains < 0] = 0
        loses[loses > 0] = 0
        rolling_mean_gains = gains.ewm(
            span=rsi_period, adjust=False).mean().values
        rolling_mean_loses = np.absolute(loses.ewm(
            span=rsi_period, adjust=False).mean().values)
        rs = rolling_mean_gains / rolling_mean_loses
        dataset['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_24 = dataset['close'].ewm(
            span=24, adjust=False).mean().values
        ema_12 = dataset['close'].ewm(
            span=12, adjust=False).mean().values
        dataset['MACD'] = ema_24 - ema_12
        dataset['MACD_Signal'] = dataset['MACD'].ewm(
            span=9, adjust=False).mean().values
        dataset['MACD_Histogram'] = dataset['MACD'] - dataset['MACD_Signal']

        # VWAP
        dataset_list = dataset.to_dict('records')
        new_dataset_list = []
        last_date = None
        cummulative_pv = 0
        cummulative_volume = 0
        for item in dataset_list:
            current_date = item['date']
            if current_date != last_date:
                cummulative_pv = 0
                cummulative_volume = 0
            cummulative_pv += (item['close'] + item['high'] + item['low']) / 3
            cummulative_volume += item['volume']
            item['VWAP'] = 0
            if cummulative_volume > 0:
                item['VWAP'] = cummulative_pv / cummulative_volume
            new_dataset_list.append(item)
            last_date = current_date

        # Convert back to dataframe
        dataset = pd.DataFrame(new_dataset_list)

        # Define target variables
        print('Defining the targets.')
        dataset['target_high'] = dataset['high'].rolling(
            window=self.window).max().shift(-self.window)
        dataset['target_low'] = dataset['low'].rolling(
            window=self.window).min().shift(-self.window)
        dataset['target_close'] = dataset['close'].shift(-self.window)
        return dataset

    def get_all_dataset(self, ticket):
        print('Getting the data for the ticket')
        raw_data = self.get_raw_data(ticket)
        dataset = self.transform_data(ticket, raw_data)
        # Drop NA for Linear Regression
        dataset.dropna(inplace=True)

        features_columns = [col for col in dataset.columns if col not in [
            'timestamp', 'target_high', 'target_low', 'target_close', 'date']]
        dataset = dataset.sample(frac=1, ignore_index=True)
        target_high = dataset['target_high'].tolist()
        target_low = dataset['target_low'].tolist()
        features_df = dataset[features_columns].copy()
        return (features_df, target_high, target_low)

    def get_prediction_features(self, ticket):
        print('Getting the data for the ticket')
        raw_data = [self.get_recent_data(ticket)]
        dataset = self.transform_data(ticket, raw_data)

        features_columns = [col for col in dataset.columns if col not in [
            'timestamp', 'target_high', 'target_low', 'target_close', 'date']]
        features_df = dataset[features_columns].copy()
        features_df = features_df.tail(1)
        return features_df

    def test_train_split(self, ticket):
        print('Getting the data for the ticket')
        raw_data = self.get_raw_data(ticket)
        dataset = self.transform_data(ticket, raw_data)
        print(f'# Samples with NaN: {dataset.shape[0]}')

        # Drop NA for Linear Regression
        dataset.dropna(inplace=True)
        print(f'# Samples without NaN: {dataset.shape[0]}')

        # Define timestamp ranges for datasets
        two_weeks_ago = datetime.now() - timedelta(days=14)
        # week_ago = datetime(2022, 10, 18)
        print(f'Two weeks ago: {two_weeks_ago}')

        # Split the datasets
        features = [col for col in dataset.columns if col not in [
            'timestamp', 'target_high', 'target_low', 'target_close', 'date']]

        print('Defining the training data.')
        train_df = dataset[dataset['timestamp'] <= two_weeks_ago].copy()
        # Sample the dataset to add a little bit of randomness before training
        train_df = train_df.sample(frac=1, ignore_index=True)
        target_high_train = train_df['target_high'].tolist()
        target_low_train = train_df['target_low'].tolist()
        features_train_df = train_df[features].copy()

        print('Defining the test data.')
        test_df = dataset[dataset['timestamp'] >= two_weeks_ago].copy()
        test_df.dropna(subset=['target_high', 'target_low'], inplace=True)
        target_high_test = test_df['target_high'].tolist()
        target_low_test = test_df['target_low'].tolist()
        features_test_df = test_df[features].copy()

        return (
            features_train_df, target_high_train, target_low_train,
            features_test_df, target_high_test, target_low_test
        )

    def test_val_train_split(self, ticket):
        print('Getting the data for the ticket')
        raw_data = self.get_raw_data(ticket)
        dataset = self.transform_data(ticket, raw_data)
        # Drop NA for Linear Regression
        dataset.dropna(inplace=True)

        # Define timestamp ranges for datasets
        week_ago = datetime.now() - timedelta(days=7)
        # week_ago = datetime(2022, 10, 18)
        three_weeks_ago = datetime.now() - timedelta(days=21)
        # two_weeks_ago = datetime(2022, 10, 11)
        print(f'A week ago: {week_ago}')

        # Split the datasets
        features = [col for col in dataset.columns if col not in [
            'timestamp', 'target_high', 'target_low', 'target_close', 'date']]

        print('Defining the training data.')
        train_df = dataset[dataset['timestamp'] <= three_weeks_ago].copy()
        # Sample the dataset to add a little bit of randomness before training
        train_df = train_df.sample(frac=1, ignore_index=True)
        target_high_train = train_df['target_high'].tolist()
        target_low_train = train_df['target_low'].tolist()
        features_train_df = train_df[features].copy()

        print('Defining the validation data.')
        val_df = dataset[((dataset['timestamp'] > three_weeks_ago) & (
            dataset['timestamp'] < week_ago))].copy()
        target_high_val = val_df['target_high'].tolist()
        target_low_val = val_df['target_low'].tolist()
        features_val_df = val_df[features].copy()

        print('Defining the test data.')
        test_df = dataset[dataset['timestamp'] >= week_ago].copy()
        test_df.dropna(subset=['target_high', 'target_low'], inplace=True)
        target_high_test = test_df['target_high'].tolist()
        target_low_test = test_df['target_low'].tolist()
        target_close_test = test_df['target_close'].tolist()
        features_test_df = test_df[features].copy()

        return (
            features_train_df, target_high_train, target_low_train,
            features_val_df, target_high_val, target_low_val,
            features_test_df, target_high_test, target_low_test, target_close_test
        )
