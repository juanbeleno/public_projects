#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 20:28:13 2022
@author: Juan Beleño
"""
import os


class DayTradingFiles():
    main_directory: str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)))
    input_directory: str = os.path.join(main_directory, 'data', 'inputs')
    processed_directory: str = os.path.join(
        main_directory, 'data', 'processed')
    output_directory: str = os.path.join(main_directory, 'data', 'outputs')

    # Processed files
    ticket_candidates: str = os.path.join(
        processed_directory, 'ticket_candidates.csv')
    long_tickets_metadata: str = os.path.join(
        processed_directory, 'long_tickets_metadata.csv')
    short_tickets_metadata: str = os.path.join(
        processed_directory, 'short_tickets_metadata.csv')
    long_watchlist: str = os.path.join(
        processed_directory, 'long_watchlist.json')
    short_watchlist: str = os.path.join(
        processed_directory, 'short_watchlist.json')
    bets_metadata: str = os.path.join(processed_directory, 'bets_metadata.csv')

    # Output files
    features_sample: str = os.path.join(
        output_directory, 'features_sample.csv')
    true_positives: str = os.path.join(output_directory, 'true_positives.csv')
    false_positives: str = os.path.join(
        output_directory, 'false_positives.csv')
    false_negatives: str = os.path.join(
        output_directory, 'false_negatives.csv')
