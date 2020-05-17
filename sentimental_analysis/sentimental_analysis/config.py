#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file.

Created on Sat May 16 20:48:21 2020
@author: Juan Beleño
"""
import os


class SentimetalAnalisysConfig:
    banks = {
      'Colombia': {
          'Bancolombia': 'Bancolombia',
          'Banco de Bogotá': 'BancodeBogota',
          'Banco de Occidente': 'Bco_Occidente',
          'Banco AV Villas': 'BancosAval',
          'Davivienda': 'Davivienda',
          'Colpatria': 'ScotiaColpatria',
          'Nequi': 'Nequi',
          'RappiPay': 'RappiPay',
          'DaviPlata': 'DaviPlata'
      },
      'Brasil': {
          'Banco do Brasil': 'BancodoBrasil',
          'Nubank': 'nubank',
          'Caixa Econômica Federal': 'Caixa',
          'Banco Santander': 'santander_br',
          'Itaú': 'itau',
          'Bradesco': 'Bradesco'
      }
    }
    """
    banks = {
        'Colombia': {'Bancolombia': 'Bancolombia'}
    }
    """

    # Directories
    main_directory:str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)))
    inputs_directory: str = os.path.join(
        main_directory, 'assets', 'inputs')
    processed_directory: str = os.path.join(
        main_directory, 'assets', 'processed')

    # Files
    raw_tweets_filepath = os.path.join(inputs_directory, 'raw_tweets.csv')
