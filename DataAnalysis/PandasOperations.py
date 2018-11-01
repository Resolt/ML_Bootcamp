import pandas as pd

df = pd.DataFrame(
	{
		'col1': [1, 2, 3, 4],
		'col2': [444, 555, 666, 444],
		'col3': ['abc', 'def', 'ghi', 'xyz']
	}
)

print(df)
print(df.head(2))

# UNIQUE VALUES
print(df['col2'].unique())
print(df['col2'].nunique())
print(df['col2'].value_counts())

# CONDITIONAL SELECTION
print(df[df['col1'] > 2])

# APPLY
def timesTwo(x):
	return x * 2
	
print(df['col2'].apply(timesTwo))
print(df['col1'].apply(lambda x: x * 2))

# SOME COLUMN WORKING
print(df.drop('col2', axis=1))

for c in df.columns:
	print(df[c])

print(df.index)

# SORTING
print(df.sort_values('col2'))
print(df.sort_values('col2').reset_index())

# FIND OPERATIONS
print(df.isnull())

data = {
	'A': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
	'B': ['one', 'one', 'two', 'two', 'one', 'one'],
	'C': ['x', 'y', 'x', 'y', 'x', 'y'],
	'D': [1, 3, 2, 5, 4, 1]
}

df = pd.DataFrame(data)
print(df)

# PIVOT
piv = df.pivot_table(values='D', index=['A', 'B'], columns='C')
print(piv)
print(piv.iloc[0, 0])


