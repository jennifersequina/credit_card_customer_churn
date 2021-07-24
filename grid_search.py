import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from pre_processing import pre_processing
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# decision tree params:
dt_max_depth = [2, 3, 4, 5, None]
dt_min_samples_split = [2, 5, 10]
dt_max_features = ['auto', 'sqrt', 'log2', None]
params_df = pd.DataFrame(itertools.product(dt_max_depth, dt_min_samples_split, dt_max_features),
                            columns=['max_depth', 'min_samples_split', 'max_features'])
params_df.replace({np.nan: None}, inplace=True)

# random forest params
rf_n_estimators = [100, 500, 1000]
rf_max_depth = [2, 3, 4, 5, None]
rf_min_samples_split = [2, 5, 10]
rf_max_features = ['auto', 'sqrt', 'log2', None]
params_df = pd.DataFrame(itertools.product(rf_n_estimators, rf_max_depth, rf_min_samples_split, rf_max_features),
                            columns=['n_estimators','max_depth', 'min_samples_split', 'max_features'])
params_df.replace({np.nan: None}, inplace=True)


df = pd.read_csv('BankChurners.csv')
df2 = pre_processing(df)

X = df2.drop('attrition', axis=1)
Y = df2['attrition']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
score_list = list()
model_type = 'rf' #dt
for index, row in params_df.iterrows():
    if model_type == 'dt':
        model = DecisionTreeClassifier(max_depth=row['max_depth'],
                                   min_samples_split=row['min_samples_split'],
                                   max_features=row['max_features'])
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=row['n_estimators'],
                                   max_depth=row['max_depth'],
                                   min_samples_split=row['min_samples_split'],
                                   max_features=row['max_features'])

    score = pd.DataFrame(cross_validate(model,
                           x_train,
                           y_train,
                           scoring='recall',
                           cv=5,
                           return_estimator=True))['test_score'].mean()
    print(f"Model: {model_type} Iteration no. {index}: ")
    # print(f"Params: max_depth: {row['max_depth']}, min_samples_split: {row['min_samples_split']}, max_features: {row['max_features']}")
    print(f"Score: {score}")
    score_list.append(score)

params_df['scores'] = score_list
params_df.sort_values('scores', ascending=False)


#      n_estimators max_depth  min_samples_split max_features    scores
# 51            100      None                  2         None  0.266097
# 111           500      None                  2         None  0.263762
# 115           500      None                  5         None  0.260661
# 175          1000      None                  5         None  0.258330
# 171          1000      None                  2         None  0.257566
