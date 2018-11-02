import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cufflinks as cf
from plotly.offline import download_plotlyjs,plot,iplot#,init_notebook_mode

cf.go_offline()

df = pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())
print(df.head())

df2 = pd.DataFrame({'Category':['A', 'B', 'C'], 'Values':[32, 43, 50]})

df.iplot(kind='scatter',x='A',y='B')
plt.show()

#SHIT DOESN'T WORK - ACCOUNT IS NEEDED - FORGET IT

