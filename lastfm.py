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
lastfm['Date'] = pd.to_datetime(lastfm['Date'])
DotW = lastfm['Date'].dt.weekday_name
lastfm['Day of the Week'] = DotW
lastfm['Season'] = lastfm.apply(get_season, axis=1)
lastfm = lastfm.dropna()
lastfm = lastfm[lastfm.Date.dt.year >= 2010]

columns = ['Month', 'count of plays in month', 'day', 'plays on date']

time1 = lastfm['Date'].dt.month
time2 = lastfm['Date'].dt.date

time  = pd.DataFrame(time2.value_counts())

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

time.sort_index(inplace=True)

plt.plot(time.index, time.Date)