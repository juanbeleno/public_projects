#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collecting data from Twitter.

Created on Sat May 16 20:17:40 2020
@author: Juan Beleño
"""
from .config import SentimetalAnalisysConfig
from time import sleep
import os
import pandas as pd
import re
import tweepy


def get_tw_api():
    """Get the Twitter API object"""
    consumer_key = os.environ['TW_API_KEY']
    consumer_secret = os.environ['TW_API_SECRET_KEY']
    access_token = os.environ['TW_ACCESS_TOKEN']
    access_token_secret = os.environ['TW_ACCESS_TOKEN_SECRET']

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api


def collect_tweets(config=SentimetalAnalisysConfig()):
    """Using the Search API to Tweets about banks in Colombia and Brazil"""
    api = get_tw_api()
    tweet_data = []
    num_tweets = 100
    num_tries = 5

    for country in config.banks:
        for bank_name in config.banks[country]:
            # Choose a language based on the country
            lang = 'es'
            if country == 'Brasil':
                lang = 'pt'

            # Create the search query and some config to kindly
            # crawl data
            tw_account = config.banks[country][bank_name]
            search_query = '"{0}" AND -filter:retweets AND -filter AND -from:{1}'.format(
                bank_name, tw_account)
            tweet_counter = 0
            try_counter = 0
            unique_users = []
            unique_tweets = []
            max_id = 0
            # Iterate until get num_tweets
            while tweet_counter < num_tweets and try_counter < num_tries:
                # Gently, crawl tweets from the Twitter API. Filtering out
                # retweets and replies
                sleep(3)
                try_counter = try_counter + 1
                print('Try #{0} with {1} tweets'.format(try_counter, tweet_counter))
                try:
                    if tweet_counter == 0:
                        tweets = api.search(
                            q=search_query, count=num_tweets, lang=lang,
                            tweet_mode="extended")
                    else:
                        tweets = api.search(
                            q=search_query, count=num_tweets, lang=lang,
                            tweet_mode="extended", max_id=max_id)
                except BaseException as e:
                    print('We failed to collect data about {}.'.format(bank_name))
                    print('Error: {}'.format(e))
                    tweets = []

                # Save tweets for each bank
                for tweet in tweets:
                    # Only save one tweet per user and bank, and
                    # only save distinct tweets. Some journals have
                    # A share button that shares the same text.
                    user = tweet.user.screen_name
                    if user not in unique_users and tweet.full_text not in unique_tweets:
                        tweet_data.append({
                            'id': tweet.id,
                            'text': tweet.full_text,
                            'created_at': tweet.created_at,
                            'bank_name': bank_name,
                            'country': country
                        })
                        unique_users.append(user)
                        unique_tweets.append(tweet.full_text)
                        max_id = tweet.id
                        tweet_counter = tweet_counter + 1
            print('Info about {} was sucessfully collected.'.format(bank_name))

    # Converting the list of dictionaries in a DataFrame to save a CSV with tweets
    tweets_df = pd.DataFrame(tweet_data)
    tweets_df.to_csv(config.raw_tweets_filepath, index=False)
