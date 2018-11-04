import pandas as pd
import numpy as np
headers = ["Artist", "Album", "Song", "DateTime"]
lastfm = pd.read_csv('umarsabir.csv', names = headers, parse_dates=[3])
def get_season(row):
    if row['Date'].month >= 3 and row['Date'].month <= 5:
        return '1'
    elif row['Date'].month >= 6 and row['Date'].month <= 8:
        return '2'
    elif row['Date'].month >= 9 and row['Date'].month <= 11:
        return '3'
    else:
        return '4'
temp = pd.DatetimeIndex(lastfm['DateTime'])
lastfm['Date'] = temp.date
lastfm['Time'] = temp.time
del lastfm['DateTime']