import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

PCA_flag = 0
PCA_dim = 256
reg_dim = 369
create_submission_flag = 0

print('loading data')
df_train_X = pd.read_csv('train_X.csv')
df_train_y = pd.read_csv('train_y.csv')
y_train = df_train_y.values.flatten()
X_train = (df_train_X.drop(['business_id','user_id'], axis = 1)).values
df_train_X = pd.read_csv('val_X.csv')
df_train_y = pd.read_csv('val_y.csv')
y_val = df_train_y.values.flatten()
X_val = (df_train_X.drop(['business_id','user_id'], axis = 1)).values
y_avg = np.mean(y_train, axis=0)

if PCA_flag:
    print(X_train.shape)
    pca = PCA(n_components=256)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)
    print(X_train.shape)

reg = MLPRegressor(early_stopping=True, hidden_layer_sizes=(64,32,16,8), random_state=1, verbose=1, activation='relu', learning_rate_init=0.001, alpha=0.00001, solver='sgd')
#reg = RandomForestRegressor(max_depth=30, random_state=0, n_estimators=2, verbose=1)
#reg

reg.fit(X_train,y_train)

print('evaluating model')
y_pred = np.clip(reg.predict(X_val),0,5)

explained_variance_score = metrics.explained_variance_score(y_val, y_pred)
mean_absolute_error = metrics.mean_absolute_error(y_val, y_pred)
mean_squared_error = metrics.mean_squared_error(y_val, y_pred)
mean_squared_log_error = metrics.mean_squared_log_error(y_val, y_pred)
median_absolute_error = metrics.median_absolute_error(y_val, y_pred)
r2_score = metrics.r2_score(y_val, y_pred)

print('explained_variance_score: ' + str(explained_variance_score))
print('mean_absolute_error: ' + str(mean_absolute_error))
print('mean_squared_error: ' + str(mean_squared_error))
print('mean_squared_log_error: ' + str(mean_squared_log_error))
print('median_absolute_error: ' + str(median_absolute_error))
print('r2_score: ' + str(r2_score))

if create_submission_flag:
    df_test = pd.DataFrame.from_csv('test_X.csv')
    X_test = (df_test.drop(['business_id','user_id'], axis = 1)).values

    y_pred=[]
    for i in range(X_test.shape[0]):
        if (np.any(np.isnan([X_test[i,:]]))==False):
            if PCA_flag:
                X_in = pca.transform(np.array([X_test[i,:]]))[0]
                tmp = reg.predict(np.reshape(X_in,(1,PCA_dim)))
                y_pred.append(tmp)
            else:
                tmp = reg.predict(np.reshape(X_test[i,:],(1,reg_dim)))
                y_pred.append(tmp)
        else:
            y_pred.append(y_avg)
    y_pred = np.clip(y_pred,0,5)
    y_pred = pd.DataFrame(y_pred, columns=['stars'])
    y_pred.to_csv('y_test.csv')

    subprocess.call(['chmod +x format.sh'])
    subprocess.call(['./format.sh'])
