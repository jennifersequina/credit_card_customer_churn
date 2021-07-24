import pandas as pd
import numpy as np
from pre_processing import pre_processing
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from manual_validation import validate
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('BankChurners.csv')
df2 = pre_processing(df)

X = df2.drop('attrition', axis=1)
Y = df2['attrition']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

rfc = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features=None)

rfc.fit(x_train, y_train)
rfc_y_pred = rfc.predict(x_test)
rfc_score = np.round(rfc.score(x_test, y_test), 2)
rfc_p_score = np.round(precision_score(y_test, rfc_y_pred, average='binary'), 2)
rfc_r_score = np.round(recall_score(y_test, rfc_y_pred, average='binary'), 2)
rfc_f_score = np.round(f1_score(y_test, rfc_y_pred, average='binary'), 2)
rfc_confusion = confusion_matrix(y_test, rfc_y_pred, labels = None)

print('Accuracy score of Random Forest Classifier is :', rfc_score)
print('Precision score of Random Forest Classifier is :', rfc_p_score)
print('Recall score of Random Forest Classifier is :', rfc_r_score)
print('F1 score of Random Forest Classifier is :', rfc_f_score)
print('Confusion Matrix of Random Forest Classifier is :')
print(rfc_confusion)
#
# Accuracy score of Random Forest Classifier is : 0.85
# Precision score of Random Forest Classifier is : 0.57
# Recall score of Random Forest Classifier is : 0.27
# F1 score of Random Forest Classifier is : 0.37
# Confusion Matrix of Random Forest Classifier is :
# [[1643   66]
#  [ 230   87]]

# double checking using manual_validation:
rfc_metric_value = validate(test_df=x_test, test_labels=y_test, ml_model=rfc, metric_name='precision', confusion_matrix=False)

# get the top important features
importance_df = pd.DataFrame(list(zip(
    x_train.columns.to_list(),
    list(rfc.feature_importances_))), columns=['feature','importance']).sort_values('importance', ascending=False)
importance_df.head(10)

#                feature  importance
# 7               income    0.131522
# 4     contacts_in_year    0.130231
# 8         credit_limit    0.114540
# 9          txn_in_year    0.087560
# 5   number_of_products    0.079411
# 1       marital_status    0.079083
# 2      education_level    0.079006
# 6      dependent_count    0.071877
# 10     avg_utilization    0.061345
# 0               gender    0.046692


# plotting confusion matrix using matplotlib
# get and reshape confusion matrix data
matrix = confusion_matrix(y_test, rfc_y_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# build the plot
plt.figure(figsize=(10,7))
sns.set(font_scale=1)
sns.heatmap(matrix, annot=True, annot_kws={'size':8},
            cmap=plt.cm.Blues, linewidths=0.2)

# add labels to the plot
class_names = ['Attrited (1)', 'Existing (0)']
tick_marks = np.arange(len(class_names)) + 0.5
tick_marks2 = tick_marks
plt.xticks(tick_marks, class_names, rotation=0)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()

