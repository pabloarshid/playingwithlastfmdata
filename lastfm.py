import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

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
ax1.plot(timedate['Date'].rolling(30).mean(), 'b',label= 'MA 30 days')

ax2 = fig.add_subplot(232)
ax2.bar(timeyear.index, timeyear.Date)

ax3 = fig.add_subplot(233)
ax3.bar(timemonth.index, timemonth.Date)
ax3.plot(timemonth['Date'].rolling(3).mean(),'b',label= 'MA 3 Months')

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


fig = plt.figure()
ax1 = fig.add_subplot(121)
TOD = pd.DataFrame(lastfm['TOD'].value_counts())
idx=TOD.index.tolist()
x = range(len(idx))
plt.bar(x, TOD['TOD'].values)
plt.xticks(x, idx)

specificTime = pd.DataFrame(lastfm['DateTime'].dt.hour.value_counts())
ax2 = fig.add_subplot(122)
plt.bar(specificTime.index, specificTime.DateTime.values)


fig.savefig('AMPM and hourly totals.png')

am = lastfm[lastfm['TOD'] == 'AM']
pm = lastfm[lastfm['TOD'] == 'PM']

amVC = pd.DataFrame(am['Day of the Week'].value_counts())
pmVC = pd.DataFrame(pm['Day of the Week'].value_counts())

ammonthlyVC = pd.DataFrame(am['Date'].dt.month.value_counts())
pmmonthlyVC = pd.DataFrame(pm['Date'].dt.month.value_counts())

amyearlyVC = pd.DataFrame(am['Date'].dt.year.value_counts())
pmyearlyVC = pd.DataFrame(pm['Date'].dt.year.value_counts())

amSeasonVC = pd.DataFrame(am['Season'].value_counts())
pmSeasonVC = pd.DataFrame(pm['Season'].value_counts())

fig = plt.figure()

ax1 = fig.add_subplot(221)
idy=DoftheW.index.tolist()
y = range(len(idy))
plt.bar(y, amVC['Day of the Week'].values, color='b',align='center')
plt.bar(y, pmVC['Day of the Week'].values, color='g',align='center')
plt.xticks(y, idy, rotation=90)
ax1.legend( ('AM', 'PM') )

ax2 = fig.add_subplot(222)
idy=Season.index.tolist()
y = range(len(idy))
plt.bar(y, amSeasonVC['Season'].values, color='b',align='center')
plt.bar(y, pmSeasonVC['Season'].values, color='g',align='center')
plt.xticks(y, idy, rotation=90)
ax2.legend( ('AM', 'PM') )


ax3 = fig.add_subplot(223)
idy=timemonth.index.tolist()
y = range(len(idy))
plt.bar(y, ammonthlyVC['Date'].values, color='b',align='center')
plt.bar(y, pmmonthlyVC['Date'].values, color='g',align='center')
plt.xticks(y, idy)
ax3.legend( ('AM', 'PM') )


ax4 = fig.add_subplot(224)
idy=timeyear.index.tolist()
y = range(len(idy))
plt.bar(y, amyearlyVC['Date'].values, color='b',align='center')
plt.bar(y, pmyearlyVC['Date'].values, color='g',align='center')
plt.xticks(y, idy)
ax4.legend( ('AM', 'PM') )
fig.savefig('AMPM Comparison.png')


lastfm['Month'] = lastfm['Date'].dt.month
lastfm['Year'] = lastfm['Date'].dt.year

dotw = lastfm.groupby('Day of the Week').Year.value_counts()
dotw.to_frame().Year
dotw = dotw.reset_index(level=["Day of the Week"])
dotw = dotw.rename(columns={'Year': 'Count'})
dotw.reset_index(level=0, inplace=True)

sns_plot = sns.catplot("Day of the Week","Count", "Year", data=dotw, kind="bar", palette="muted",height=8.27, aspect=11.7/8.27)
sns_plot.savefig("DayofWeek vs Year count Seaborn.png")

dotwmonth = lastfm.groupby('Day of the Week').Month.value_counts()
dotwmonth.to_frame().Month
dotwmonth = dotwmonth.reset_index(level=["Day of the Week"])
dotwmonth = dotwmonth.rename(columns={'Month': 'Count'})
dotwmonth.reset_index(level=0, inplace=True)
sns_plot = sns.catplot("Month","Count","Day of the Week", data=dotwmonth, kind="bar", palette="muted",height=8.27, aspect=11.7/8.27)
sns_plot.savefig("DayofWeek vs Month count Seaborn.png")


doampm = lastfm.groupby('Day of the Week').TOD.value_counts()
doampm.to_frame().TOD
doampm = doampm.reset_index(level=["Day of the Week"])
doampm = doampm.rename(columns={'TOD': 'Count'})
doampm.reset_index(level=0, inplace=True)
sns_plot = sns.catplot("Day of the Week","Count", "TOD", data=doampm, kind="bar", palette="muted",height=8.27, aspect=11.7/8.27)
sns_plot.savefig("DayofWeek vs AMPM count Seaborn.png")


# My top 20 Artists
# Lets compare my favorite artists
artist = pd.DataFrame(lastfm['Artist'].value_counts())
artist = artist[:20]
artist.reset_index(level=0, inplace=True)
artist = artist.rename(columns={'index': 'Artist', 'Artist':'Count'})

sns_plot = sns.catplot("Artist", "Count", data=artist, kind="bar", palette="muted", height=8.27, aspect=25/8.27)
sns_plot.savefig("Top 20 Artists.png")
