#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 20:28:13 2022
@author: Juan Beleño
"""
import os

class DayTradingFiles():
    main_directory:str = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    input_directory: str = os.path.join(main_directory, 'data', 'inputs')
    output_directory: str = os.path.join(main_directory, 'data', 'outputs')

    # Inputs
    aapl_filepath: str = os.path.join(input_directory, 'AAPL_interval_5m_range_1mo.json')

    # Outputs
    high_model_filepath: str = os.path.join(output_directory, 'high_model.joblib')
    low_model_filepath: str = os.path.join(output_directory, 'low_model.joblib')
    features_filepath: str = os.path.join(output_directory, 'features.csv')
    target_high_filepath: str = os.path.join(output_directory, 'target_high.json')
    target_low_filepath: str = os.path.join(output_directory, 'target_low.json')
