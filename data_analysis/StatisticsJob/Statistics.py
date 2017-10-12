import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv("/Users/chandler/Desktop/status_test/day=20130530/part-00000-87532fcd-a31a-4bb2-acc8-fcc1330d39a8.c000.csv")
log_year = log.set_index("city")
log_year['traffic'].plot()
plt.legend(loc='best')
from pandas.tools.plotting import scatter_matrix
scatter_matrix(log, alpha=0.2, figsize=(6, 6), diagonal='kde')