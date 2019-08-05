import pandas as pd
import numpy as np

dataset = pd.read_csv("50_Startups.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#avaoiding dummy variable trap

x= x[:,1:]

# split the data \set into the training set and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2,random_state = 0)

# fitting multiple linear regression to the traininng set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set result

y_pred = regressor.predict(x_test)

# building optimal model using backward elemination

import statsmodels.formula.api as sm
# looping through the coloumns of x and eleminating the coloumn with highest p value
def bacelem(x,sl):
    numvars = len(x[0])
    for i in range(0,numvars):
        regressor_ols = sm.OLS(endog= y,exog= x_opt).fit()
        maxvar = max(regressor_ols.pvalues).astype(float)
        if maxvar > sl:
            for j in range(0,numvars - i):
                if (regressor_ols.pvalues[j].astype(float) == maxvar):
                    x = np.delete(x,j,1)
    regressor_ols.summary()
    return x

sl = 0.05
x_opt = x[:,[0, 1, 2, 3, 4]]
x_modeled = bacelem(x_opt,sl)


