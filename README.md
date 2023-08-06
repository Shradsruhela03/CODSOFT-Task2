# CODSOFT-Task2 Titanic Survival Prediction
name: shraddha ruhela
Batch: July
Domain: Data Science

# Aim
The aim of this project is to build a model that predicts whether a passenger on the Titanic survived or not based on given features.

# Dataset
The dataset for this project is imported from a CSV file, "archive.zip". The dataset contains information about passengers on the Titanic, including their survival status, class (Pclass), sex (Gender), and age (Age).

# Libraries Used
The following important libraries were used for this project:
1-numpy
2-pandas
3-matplotlib.pyplot
4-seaborn
5-sklearn.preprocessing.LabelEncoder
6-sklearn.model_selection.train_test_split
7-sklearn.linear_model.LogisticRegression

# Model Training
The feature matrix X and target vector Y were created using relevant columns from the DataFrame.
The dataset was split into training and testing sets using train_test_split from sklearn.model_selection.
A logistic regression model was initialized and trained on the training data using LogisticRegression from sklearn.linear_model.

# Model Prediction
The model was used to predict the survival status of passengers in the test set.
The predicted results were printed using log.predict(X_test).
The actual target values in the test set were printed using Y_test.
A sample prediction was made using log.predict([[2, 1]]) with Pclass=2 and Sex=Male (1).

