import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dir = 'Bootcamp/DataCapstoneProjects/'

df = pd.read_csv(dir+'911.csv')

print(df.info())
print(df.head())

# TOP 5 ZIP CODES FOR 911 CALLS
print("\nTop 5 Zip Codes for 911 calls:\n{}\n".format(df['zip'].value_counts().head(5)))

# TOP 5 TOWNSHIPS FOR 911 CALLS
print("Top 5 townships for 911 calls:\n{}\n".format(df['twp'].value_counts().head(5)))

# AMOUNT OF UNIQUE TITLES FOR THE CALLS
print("Amount of unique titles: {}\n".format(df['title'].nunique()))

# CREATE REASON COLUMS
df['reason'] = df['title'].apply(lambda x: x.split(':')[0])

# MOST COMMON REASON
print("The most common reason for 911 call: {}\n".format(df['reason'].value_counts().head(1)))

# COUNTPLOT FOR REASON
sns.set_style('darkgrid')
sns.countplot(df['reason'])
plt.show()

# DATATYPE OF TIMESTAMP COLUMN
print("Data type of the timeStamp column: {}\n".format(df['timeStamp'].dtype)) # THIS IS SUPPOSED TO OUTPUT str

# FORMAT DATETIME
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

# HOUR OF FIRST ELEMENT IN TIMESTAMP
print("Hour for first element: {}\n".format(df['timeStamp'].iloc[0].hour))

# CREATE HOUR MONTH AND DAY COLUMNS
df['hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['month'] = df['timeStamp'].apply(lambda x: x.month)
df['day'] = df['timeStamp'].apply(lambda x: x.dayofweek)

# MAP NAMES OF WEEK DAYS INSTEAD OF INTEGERS
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['day'] = df['day'].map(dmap)

# COUNPLOT OF WEEKDAYS WITH REASON AS HUE
sns.countplot(data=df, x='day', hue='reason')
plt.show()

# COUNTPLOT OF MONTHS WITH REASON AS HUE
sns.countplot(data=df, x='month', hue='reason')
plt.show()

# PLOTTING DATA FOR MISSING MONTHS
byMonth = df.groupby('month').count()
print(byMonth)
byMonth['twp'].plot()
plt.show()

sns.lmplot(x='month',y='twp',data=byMonth.reset_index())
plt.show()

# CREATE DATE COLUMN
df['date'] = df['timeStamp'].apply(lambda x: x.date())
print(df['date'].head())

# GROUP BY DATE
byDate = df.groupby('date').count()
byDate['twp'].plot()
plt.show()

# PLOT DATE FOR EACH REASON
for n in df['reason'].unique():	
	df[df['reason'] == n].groupby('date').count()['twp'].plot()
	plt.show()

# DAY HOUR DATAFRAME
dayHour = df.groupby(by=['day','hour']).count()['reason'].unstack()
print(dayHour.head())

# HEAT MAP
sns.heatmap(dayHour,cmap='plasma')
plt.show()

# CLUSTER MAP
sns.clustermap(dayHour)
plt.show()

# DAY MONTH DATA FRAME
dayMonth = df.groupby(by=['day','month']).count()['reason'].unstack()
print(dayMonth.head())

# HEAT MAP
sns.heatmap(dayMonth,cmap='plasma')
plt.show()

# CLUSTER MAP
sns.clustermap(dayMonth,cmap='plasma')
plt.show()
