import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv('./data/BlackFriday.csv')
data.fillna(0.0, inplace=True)
le = LabelEncoder()
data['User_ID'] = le.fit_transform(data['User_ID'])
data['Product_ID'] = le.fit_transform(data['Product_ID'])
data_AGCS = pd.get_dummies(data,columns=['Age','Gender', 'City_Category', 'Stay_In_Current_City_Years'])
data_encoded = pd.concat([data, data_AGCS], axis=1)
data_encoded.drop(['Age','Gender', 'City_Category', 'Stay_In_Current_City_Years'], axis=1, inplace=True)
X = data_encoded.drop('Purchase',axis=1)
y = data_encoded['Purchase']
# 标准化
std = StandardScaler()
X = std.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y)

rfr = RandomForestRegressor(n_estimators=10)
rfr.fit(X_train, y_train)
y_pre = rfr.predict(X_test)
print(y_pre)

