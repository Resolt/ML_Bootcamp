# from pandas_datareader import data, wb
import pandas as pd
import numpy as np
# import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# # TIME OBJECTS
# start = datetime.datetime(2006, 1, 1)
# end = datetime.datetime(2016, 1, 1)

# API = 'robinhood'

# # BANK TICKER DATAFRAMES
# BAC = data.DataReader("BAC", API, start, end) # BANK OF AMERICA
# C = data.DataReader("C", API, start, end) # CITIGROUP
# GS = data.DataReader("GS", API, start, end) # GODLMAN SACHS
# JPM = data.DataReader("JPM", API, start, end) # JPMORGAN CHASE
# MS = data.DataReader("MS", API, start, end) # MORGAN STANLEY
# WFC = data.DataReader("WFC", API, start, end) # WELLS FARGO

# # BANK STRING
# tickers = sorted(['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC'])

# # CONCATENATE
# bank_list = [BAC, C, GS, JPM, MS, WFC]
# for b in bank_list:
# 	b.reset_index(level=1,inplace=True,drop=True)


# bank_stocks = pd.concat(bank_list, axis=1, keys=tickers)

# # COL NAMES
# bank_stocks.columns.names = ['Bank Ticker', 'Stock Info']

# bank_stocks.to_csv('test.csv')

dir = 'Bootcamp/DataCapstoneProjects/'

bank_stocks = pd.read_pickle(dir+'all_banks')

print(bank_stocks.head())

# MAX CLOSE PRICE FOR EACH BANKS STOCK
print("Max close price for each bank:\n{}\n".format(
	bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()
))

# CREATE EMOPTY RETURNS FRAME
returns = pd.DataFrame()

tickers = sorted(['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC'])

# POPULATE RETURNS FRAME
for t in tickers:
	returns[t+' Return'] = bank_stocks.xs(key='Close',axis=1,level="Stock Info")[t].pct_change()


returns.to_csv('returns.csv')
print(returns.head())

# PAIR PLOT ON RETURNS FRAME
sns.set_style('darkgrid')
sns.pairplot(data=returns.dropna(how='all'))
plt.show()

# BEST AND WORST DATES
# for n in returns.columns:
# 	print("{}\tWorst: {}\tBest: {}".format(
# 		n.split()[0],
# 		returns[n][returns[n] == returns[n].min()].index.date[0],
# 		returns[n][returns[n] == returns[n].max()].index.date[0]
# 	))

print("Worst:\n{}\n".format(
	returns.idxmin()
))

print("Best:\n{}\n".format(
	returns.idxmax()
))

# RISKIEST BANK OF THE PERIOD
print("\nList of standard deviations:\n{}\n".format(
	returns.std()
))

# RISKIEST BANK OF 2015
print("List of standard deviations of 2015:\n{}\n".format(
	returns.ix['2015-01-01':'2015-12-31'].std()
))

# DIST PLOT FOR RETURNS OF 2015 FOR MORGAN STANLEY
sns.distplot(returns["MS Return"].ix['2015-01-01':'2015-12-31'],bins=40)
plt.show()

# DIST PLOT FOR RETURNS OF 2008 FOR CITIGROUP
sns.distplot(returns["C Return"].ix['2008-01-01':'2008-12-31'],bins=40)
plt.show()

