from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb



n_rows = 300000


df = pd.read_csv("train", nrows=n_rows)


Y = df['click'].values
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'],
            axis=1).values


n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]


enc = OneHotEncoder(handle_unknown='ignore')

X_train_enc = enc.fit_transform(X_train)

X_test_enc = enc.transform(X_test)

parameters = {'max_depth': [3, 10, None]}

decision_tree = DecisionTreeClassifier(criterion='gini',
                                       min_samples_split=30)

grid_search = GridSearchCV(decision_tree, parameters,
                           n_jobs=-1, cv=3, scoring='roc_auc')

grid_search.fit(X_train_enc, Y_train)

decision_tree_best = grid_search.best_estimator_

pos_prob = decision_tree_best.predict_proba(X_test_enc)[:, 1]

print(f'The ROC AUC on testing set is: {roc_auc_score(Y_test,pos_prob):.3f}')


