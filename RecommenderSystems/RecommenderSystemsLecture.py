import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

# READ THE DATA
dir="Bootcamp/RecommenderSystems/"
df = pd.read_csv(dir+'u.data',sep='\t',names=['user_id','item_id','rating','timestamp'])

print(df.head())
print(df.info())

# PART 1
movie_titles = pd.read_csv(dir+'Movie_Id_Titles')
print(movie_titles.head())
print(movie_titles.info())

df = pd.merge(df,movie_titles,on='item_id')
print(df.head())
print(df.info())

print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())

print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['no'] = df.groupby('title')['rating'].count()

# sns.distplot(ratings['no'],kde=False,bins=70)
# plt.show()

# sns.distplot(ratings['rating'],kde=False,bins=70)
# plt.show()

# sns.jointplot(x='rating',y='no',data=ratings)
# plt.show()

# PART 2
moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
print(moviemat.head())

print(ratings.sort_values('no',ascending=False).head(10))

sw_user_ratings = moviemat['Star Wars (1977)']
ll_user_ratings = moviemat['Liar Liar (1997)']

# STAR WARS

print(sw_user_ratings.head())

corr_sw = pd.DataFrame(moviemat.corrwith(sw_user_ratings).sort_values(ascending=False),columns=['Correlation'])
corr_sw.dropna(inplace=True)

print(corr_sw.head())

corr_sw = corr_sw.join(ratings['no'])
print(corr_sw.head())

print(corr_sw[corr_sw['no']>100].sort_values('Correlation',ascending=False).head(10))

# LIAR LIAR

corr_ll = pd.DataFrame(moviemat.corrwith(ll_user_ratings).sort_values(ascending=False),columns=['Correlation'])
corr_ll.dropna(inplace=True)

print(corr_ll.head())

corr_ll = corr_ll.join(ratings['no'])
print(corr_ll.head())

print(corr_ll[corr_ll['no']>100].sort_values('Correlation',ascending=False).head(10))