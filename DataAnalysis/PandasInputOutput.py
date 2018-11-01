import numpy as np
import pandas as pd

dir = "Bootcamp/DataAnalysis/"

df = pd.read_csv(dir+'example.csv')
print(df)
df.to_csv(dir + 'example_out.tsv', sep='\t', index=False)

df = pd.read_excel(dir + 'Excel_Sample.xlsx', sheet_name='Sheet1')
print(df)
df.to_excel(dir + 'Excel_Sample_out.xlsx', sheet_name='John')

df = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')
print(df[0])
