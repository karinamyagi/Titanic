import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
#%matplotlib inline
titan = pd.read_csv("../data/titanic.csv")
titan.head()

##Cross Tabulation between SURVIVAL RATE and CLASS
titan[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

##BarPlot of SURVIVAL RATE and CLASS
sns.barplot(x="Pclass", y="Survived", data=titan)

##Cross Tabulation between SURVIVAL RATE and GENDER
titan[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

##BarPlot of SURVIVAL RATE and GENDER
sns.barplot(x = "Sex", y = "Survived", data = titan)

#transform Age in a categorical variable
titan["Age"] = titan["Age"].fillna(-0.5)
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Baby', 'Kid', 'Teen', 'Student', 'Young Adult', 'Adult', 'Senior']
titan['AgeGroup'] = pd.cut(titan["Age"], bins, labels = labels)

##Cross Tabulation between SURVIVAL RATE and AGE
titan[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean().sort_values(by='Survived', ascending=False)

##BarPlot of SURVIVAL RATE and AGE GROUP
sns.barplot(x="AgeGroup", y="Survived", data=titan)
plt.show()