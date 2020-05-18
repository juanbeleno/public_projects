#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing strings for tweets. We are working with spanish and portuguese
texts, then we can use the winner repositories of Mercado Libre Challenge to
borrow some ideas.

Source of inspiration:
https://github.com/pablozivic/mercadolibre/blob/master/MeLi_BaseGen/MeLi_BaseGen_V3.ipynb

Created on Sun May 17 20:56:54 2020
@author: Juan Beleño
"""
from .config import SentimetalAnalisysConfig
from unicodedata import normalize


def normalize_tweet(text, config=SentimetalAnalisysConfig()):
    """Normalize texts to ease NLP tasks"""
    # Convert to lowercase
    normalized_text = text.lower()
    # Remove accents
    normalized_text = normalize('NFKD', normalized_text).encode('ascii', errors='ignore').decode('utf-8')
    # Remove extra spaces
    normalized_text = normalized_text.replace(r'[\s\n\r]+', ' ')
    # Remove some plurals in spanish and portuguese
    normalized_text = normalized_text.replace(
        '\\b([a-zA-Z]+[aeiouwy])(s)\\b', r'\1')
    # Anonymize twitter accounts
    normalized_text = normalized_text.replace(
        '\\@[a-z0-9]+\\'.format(tw_user.lower()), config.anon_tw_user)
    # Set all digits to zero
    normalized_text = normalized_text.replace('[0-9]', '0')
    # Set spaces before and after punctuation signs
    normalized_text = normalized_text.replace('([^a-z0-9])', r' \1 ')
    # Normalize links
    normalized_text = normalized_text.replace('\\bhttps?://(.+)\\b', config.anon_links)
    # Strip spaces at the beginning and end of the text
    normalized_text = normalized_text.strip(' \n\r\t')
    return normalized_text
