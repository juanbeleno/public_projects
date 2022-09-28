#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:44:46 2022
@author: Juan Beleño
"""
from datetime import datetime
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

    def prepare_datasets(self):
        raw_data = self.get_raw_data('APPL', '3mo', '5m')
        timestamps = raw_data['chart']['result']['timestamp']
        data = raw_data['chart']['result']['indicators']['quote']
        train_df = pd.DataFrame({
            'timestamp': timestamps,
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'open': data['open'],
            'volume': data['volume']
        })

        # Convert int to timestamp
        train_df['timestamp'] = train_df['timestamp'].apply(
            lambda timestamp: pd.to_datetime(timestamp, utc=True, unit='s')
        )

        # Moving features
        column = 'close'
        periods = [5, 10, 20, 30, 50]
        for period in periods:
            train_df.loc[:,'Return_{period}'] = train_df[column].pct_change(period)
            train_df.loc[:,'MovingAvg_{period}'] = train_df[column].rolling(window=period).mean().values
            train_df.loc[:,'ExpMovingAvg_{period}'] = train_df[column].ewm(span=period, adjust=False).mean().values
            train_df.loc[:,'Volatility_{period}'] = train_df[column].diff().rolling(period).std()

        # Define target variables
        train_df['target_high'] = train_df['high'].rolling(window=8).max().shift(-8)
        train_df['target_low'] = train_df['low'].rolling(window=8).max().shift(-8)
