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

def ampm(row):
    if row["DateTime"].hour < 12:
        return 'AM'
    else:
        return 'PM'

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
lastfm["TOD"] = lastfm.apply(ampm, axis=1)
lastfm['Day of the Week'] = DotW
lastfm = lastfm.dropna()
lastfm = lastfm[lastfm.Date.dt.year >= 2010]

# New dataframe for first plots

time1 = lastfm['Date'].dt.month
time2 = lastfm['Date'].dt.date
time3 = lastfm['Date'].dt.year


timemonth = pd.DataFrame(time1.value_counts())
timedate = pd.DataFrame(time2.value_counts())
timeyear = pd.DataFrame(time3.value_counts())
Season = pd.DataFrame(lastfm['Season'].value_counts())
DoftheW = pd.DataFrame(lastfm['Day of the Week'].value_counts())

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
# Set figure width to 12 and height to 9
fig_size[0] = 18
fig_size[1] = 15
plt.rcParams["figure.figsize"] = fig_size

timemonth.sort_index(inplace=True)
timedate.sort_index(inplace=True)
timeyear.sort_index(inplace=True)
Season.sort_index(inplace=True)
DoftheW.sort_index(inplace=True)

fig = plt.figure()

ax1 = fig.add_subplot(231)
ax1.bar(timedate.index, timedate.Date)

ax2 = fig.add_subplot(232)
ax2.bar(timeyear.index, timeyear.Date)

ax3 = fig.add_subplot(233)
ax3.bar(timemonth.index, timemonth.Date)

ax4 = fig.add_subplot(234)
idx=Season.index.tolist()
x = range(len(idx))
plt.bar(x, Season['Season'].values)
plt.xticks(x, idx, rotation=90)


ax5 = fig.add_subplot(235)
idy=DoftheW.index.tolist()
y = range(len(idy))
plt.bar(y, DoftheW['Day of the Week'].values)
plt.xticks(y, idy, rotation=90)

fig.savefig('combined totals.png')

del timemonth, timeyear, timedate, Season, DoftheW

