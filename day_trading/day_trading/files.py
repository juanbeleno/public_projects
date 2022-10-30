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
    processed_directory: str = os.path.join(main_directory, 'data', 'processed')
    output_directory: str = os.path.join(main_directory, 'data', 'outputs')

    # Processed files
    ticket_candidates: str = os.path.join(processed_directory, 'ticket_candidates.csv')
    tickets_metadata: str = os.path.join(processed_directory, 'tickets_metadata.csv')
    selected_tickets: str = os.path.join(processed_directory, 'selected_tickets.json')
    bets_metadata: str = os.path.join(processed_directory, 'bets_metadata.csv')
