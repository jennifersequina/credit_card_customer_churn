## Credit Card Customers Churning Prediction

### Description:
This project is about predicting the churning possibility of credit card customers using the dataset from Kaggle (credits to: Sakshi Goyal) which consists of 10,000 customers mentioning their age, salary, marital status, credit card limit, credit card category, etc.
The datasets only have 16% of customers who have churned, and some features have imbalance in the distribution. Thus, we need to try different machine learning models and tweak the parameters to get the best scores using grid search.

This package contains the 1 datasets and 4 python files:

- BankChurners.csv
- main.py
- pre_processing.py
- manual_validation.py
- grid_search.py

### Methodology:
This section explains the procedures in completing this project.

1. Analyzing each column in the datasets to check which information will be useful in predicting.
    - BankChurners.csv - most of the columns are useful in predicting churners, but we need to ignore the last two columns for naive bayes.
    - pre_processing.py - in this python file, I created a function to pre-process the datasets. I analyzed all the columns using value_counts and histogram to see the distribution, and decide which approach to use in converting columns.
      - First, I imputed the 'Unknown' values in the columns Education_Level, Income_Category & Marital_Status with the top value count from the datasets.
      - Second, I converted the most of the columns with categorical values to numerical values by creating new columns conditional on the other columns.
      - Lastly, I dropped the unnecessary columns, rename the client_id column and set it to index. It's not necessary in the splitting of datasets for train and test, but it will be useful for checking the data later on.
   
2. Building a machine learning model suitable for the objective. For classification, there's a lot of machine learning models to choose from.
   In this project, I tried Decision Tree Classifier, Logistic Regression, and Random Forest Classifier.
   - main.py & manual_validation.py
     - I created 2 dataframes inside X and Y variables, the dataframe in X has no 'attrition' column while the dataframe in Y variable has only 'attrition' column as this is my target variable.
     - Next, I split the train and test dataset, as a general rule of thumb, training data is 80% while testing data is 20%.
     - Initially, I tried fitting the train datasets in the machina learning models - Decision Tree Classifier and Logistic Random Classifier. Then check the metric scores against testing data.
     - I also created manual_validation.py to check the metric scores uch as accuracy score, precision, recall, f1 scores, confusion matrix. It is not a requirement, but I think it's good to know how to write your own function to validate and to also understand how the metrics are being computed.
     - After running the metric scores (built-in function & manual validation), I noticed that they seem not be the best model for the use case.
   - grid_search.py
      - To further check which model is better, I created grid_search python file where it will check different set of parameters using Decision Tree Classifier and Random Forest Classifier. Then do a cross validation to get the mean scores of each fold, and print the top 10 scores of the parameters set.
      - From there, I identified the best parameters to use specifically for Random Forest Classifier, then I used it in the main.py in getting the metric scores.
     
   - After getting the best results, I checked the top important features affecting the churning prediction. This information could be useful in further analyzing the data and improving the model.

NOTE: I used the grid_search multiple times until I get the highest recall score. This score is important in order to know the effectiveness of the machine learning model for this use case, as I want the model to detect as many churns as possible.


### Usage:
This package can be used to make a prediction whether the customers have the possibility of churning based on the client's information, and transactions activity. From that predictions, banks can make a preventive actions to avoid these possible churns. 

