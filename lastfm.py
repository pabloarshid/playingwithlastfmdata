import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


headers = ["Artist", "Album", "Song", "DateTime"]
lastfm = pd.read_csv('umarsabir.csv', names = headers, parse_dates=[3])

# Season is defined by month not month and and date

def get_season(row):
    if row['Date'].month >= 3 and row['Date'].month <= 5:
        return '1'
    elif row['Date'].month >= 6 and row['Date'].month <= 8:
        return '2'
    elif row['Date'].month >= 9 and row['Date'].month <= 11:
        return '3'
    else:
        return '4'

# get more specific seasons 

def get_season2(row):
    if (row['Date'].month >= 3 and row['Date'].day >= 20)  and (row['Date'].month <= 6 and row['Date'].day <= 20):
        return '1'
    elif (row['Date'].month >= 6 and row['Date'].day >= 21) and (row['Date'].month <= 9 and row['Date'].day <= 20):
        return '2'
    elif (row['Date'].month >= 9 and row['Date'].day >= 20) and (row['Date'].month <= 12 and row['Date'].day <= 20):
        return '3'
    else:
        return '4'

# Seperate Date and Time
# Add days of the week
# drop any row with date before 2010, these values mean that the date wasnt properly
# recorded since i started lastfm in 2011

temp = pd.DatetimeIndex(lastfm['DateTime'])
lastfm['Date'] = temp.date
lastfm['Time'] = temp.time
lastfm['Date'] = pd.to_datetime(lastfm['Date'])
DotW = lastfm['Date'].dt.weekday_name
lastfm['Season'] = lastfm.apply(get_season, axis=1)
lastfm['Day of the Week'] = DotW
lastfm = lastfm.dropna()
lastfm = lastfm[lastfm.Date.dt.year >= 2010]

# New dataframe for first plots

time1 = lastfm['Date'].dt.month
time2 = lastfm['Date'].dt.date

timemonth = pd.DataFrame(time1.value_counts())
timedate = pd.DataFrame(time2.value_counts())

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

timemonth.sort_index(inplace=True)
timedate.sort_index(inplace=True)

plt.xlabel('Months')
plt.ylabel('Total Plays')
plt.plot(timemonth.index, timemonth.Date)
plt.savefig('Total plays per month Plot.png')
plt.close()

plt.xlabel("Day")
plt.ylabel('Total Plays')
plt.plot(timedate.index, timedate.Date)
plt.savefig('Total plays Per day Plot.png')
plt.close()

