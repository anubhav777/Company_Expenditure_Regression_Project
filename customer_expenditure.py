import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df=pd.read_csv('Ecommerce Customers')
df.head()
df.describe()
df.info()

sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=df)

sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=df)

sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=df,kind="hex")

sns.pairplot(df)

sns.lmplot(x='Yearly Amount Spent',y='Length of Membership',data=df)

X = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership','Yearly Amount Spent']]

y = df['Yearly Amount Spent']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

lm=LinearRegression()

lm.fit(X_train,y_train)

print(lm.coef_)

pred = lm.predict(X_test)

sns.scatterplot(y_test,pred)


MAE=metrics.mean_absolute_error(y_test,pred)
MSE=metrics.mean_squared_error(y_test,pred)
RMSE=np.sqrt(metrics.mean_squared_error(y_test,pred))

print(MAE)
print(MSE)
prnt(RMSE)

sns.distplot((y_test - pred))

pred = lm.coef_
new_df = pd.DataFrame(g,X.columns)
new_df






