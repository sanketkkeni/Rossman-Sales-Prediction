# Sanket Keni's Rossman Sales Prediction-Kaggle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import operator
import matplotlib
import matplotlib.pyplot as plt




# Gather some features
def build_features(features, data):
    # remove NaNs
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1
    # Use some properties directly
    features.extend(['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday'])

    # Label encode some features
    features.extend(['StoreType', 'Assortment', 'StateHoliday'])
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    features.extend(['DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear'])
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.weekofyear

    # CompetionOpen en PromoOpen from https://www.kaggle.com/ananya77041/rossmann-store-sales/randomforestpython/code
    # Calculate time competition open time in months
    features.append('CompetitionOpen')
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
        (data.Month - data.CompetitionOpenSinceMonth)
    # Promo open time in months
    features.append('PromoOpen')
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
        (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0

    # Indicate that sales on that day are in promo interval
    features.append('IsPromoMonth')
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    data['monthStr'] = data.Month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1

    return data


## Start of main script

print("Load the training, test and store data using pandas")
types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(float),
         'PromoInterval': np.dtype(str)}
train = pd.read_csv("C:\\Users\\Sanket Keni\\Desktop\\Genesis\\Rossman Sales\\train.csv", parse_dates=[2], dtype=types)
test = pd.read_csv("C:\\Users\\Sanket Keni\\Desktop\\Genesis\\Rossman Sales\\test.csv", parse_dates=[3], dtype=types)
store = pd.read_csv("C:\\Users\\Sanket Keni\\Desktop\\Genesis\\Rossman Sales\\store.csv")

print("Assume store open, if not provided")
train.fillna(1, inplace=True)
test.fillna(1, inplace=True)

print("Consider only open stores for training. Closed stores wont count into the score.")
train = train[train["Open"] != 0]
print("Use only Sales bigger then zero. Simplifies calculation of rmspe")
train = train[train["Sales"] > 0]

print("Join with store")
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features = []

print("augment features")
build_features(features, train)
build_features([], test)
print(features)
print('training data processed')


# Split train-validation set
X_train, X_valid = train_test_split(train, test_size=0.012, random_state=10)
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)

def rmspe(y, yhat):
    return np.sqrt(np.mean(((y-yhat)/y) ** 2))


#################### xgBoost Model
'''
eta - step size shrinkage used in update to prevents overfitting. After each 
boosting step, we can directly get the weights of new features. and eta actually 
shrinks the feature weights to make the boosting process more conservative.

max_depth - maximum depth of a tree, increase this value will make the model more 
complex / likely to be overfitting. 0 indicates no limit, limit is required for 
depth-wise grow policy.

subsample- subsample ratio of the training instance
'''
params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.3,
          "max_depth": 12,
          "subsample": 0.85,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1301
          }
num_boost_round = 170 #number of iterations

# Used in xgb.train for evaluation
def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label()) #get_label() - Get the label of the DMatrix.
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)


'''
Both xgboost and GBM follows the principle of gradient boosting. There are however,
 the difference in modeling details. Specifically, xgboost used a more regularized 
 model formalization to control over-fitting, which gives it better performance.
 '''
# As required by xgb.train
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
print("Train a XGBoost model")
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)

print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
error = rmspe(X_valid.Sales.values, np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))
#RMSPE - Root Mean Square Percentage Error 

test['StateHoliday'] = test['StateHoliday'].astype('int32')
print("Make predictions on the test set")
dtest = xgb.DMatrix(test[features])
test_probs = gbm.predict(dtest)

### Submit on Kaggle
result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(test_probs)})
result = result.sort_values(['Id'], ascending=[True])
result.to_csv("C:\\Users\\Sanket Keni\\Desktop\\Genesis\\Rossman Sales\\out.csv", index=False)
######

# XGB feature importances
# Based on https://www.kaggle.com/mmueller/liberty-mutual-group-property-inspection-prediction/xgb-feature-importance-python/code

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

create_feature_map(features)
importance = gbm.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)
################################################################################










################# Random Forests
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=10, verbose = 1)

rfr.fit(X_train[features], y_train)
y_pred = rfr.predict(X_valid[features])


print("Validating")
error = rmspe(np.expm1(y_pred), np.expm1(y_valid))
print('RMSPE for Validation Set: {:.6f}'.format(error))


test_pred = rfr.predict(test[features])

### Submit on Kaggle
result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(test_pred)})
result = result.sort_values(['Id'], ascending=[True])
result.to_csv("C:\\Users\\Sanket Keni\\Desktop\\Genesis\\Rossman Sales\\out.csv", index=False)
#######






######### Decision Trees
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth=4, random_state=42)
dtr.fit(X_train[features], y_train)



print("Validating")
y_pred = dtr.predict(X_valid[features])
error = rmspe(np.expm1(y_pred), np.expm1(y_valid))
print('RMSPE for Validation Set: {:.6f}'.format(error))

test_pred = dtr.predict(test[features])
#############################################



################ Adaboost Regressoion
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
abr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=42)
abr.fit(X_train[features], y_train)

print("Validating")
y_pred = abr.predict(X_valid[features])
error = rmspe(np.expm1(y_pred), np.expm1(y_valid))
print('RMSPE for Validation Set: {:.6f}'.format(error))

test_pred = abr.predict(test[features])
######################################################




















