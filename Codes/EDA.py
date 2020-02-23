# Libraries ========================================================================================================== #
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')
# Data reading ======================================================================================================= #
path = "/Users/Fox/Desktop/Data Science/Capstone"
df = pd.read_csv(path + "/Codes/DataPre.csv")

# Histograms ========================================================================================================= #
# individual
sns.set(color_codes=True)
for COL in df.columns:
    hist = sns.distplot(df[COL], label=COL)
    plt.show()
    figure = hist.get_figure()
    figure.savefig(path + '/img/EDA/Hist_individual/' + COL, dpi=400)

# overview
Header = df.columns
Header = np.reshape(Header, (4, -1))
fig, ax = plt.subplots(4, 5)
for i in range(4):
    for j in range(5):
        ax[i, j].hist(df[Header[i, j]])
        ax[i, j].set_title(Header[i, j])
fig.set_size_inches(20, 14)
plt.savefig(path + '/img/EDA/Histogram.png')
fig.show()

# Correlation Plot =================================================================================================== #
corr = df.corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.savefig(path + '/img/EDA/Correlation.png')
plt.show()

# Scatter Plot ======================================================================================================= #
fig, ax = plt.subplots(4, 5)
for i in range(4):
    for j in range(5):
        ax[i, j].scatter(df[Header[i, j]], df['AdoptionSpeed'])
        ax[i, j].set_title(Header[i, j])
fig.set_size_inches(20, 14)
plt.savefig(path + '/img/EDA/Scatter.png')
fig.show()

# Box Plot =========================================================================================================== #
fig, ax = plt.subplots(4, 5)
for i in range(4):
    for j in range(5):
        ax[i, j].boxplot(df[Header[i, j]])
        ax[i, j].set_title(Header[i, j])
fig.set_size_inches(20, 14)
plt.savefig(path + '/img/EDA/Box.png')
fig.show()