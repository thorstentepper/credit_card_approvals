# Project Title
Predicting Credit Card Approvals


## Description
This project uses supervised machine learning to build a predictor for credit card approvals. The dataset is preprocessed by addressing differences in data type and scale as well as missing entries. Separate datasets for training and testing are created and a logistic regression model is fit to the train set. Finally, predictions are made based on the test set and model performance is evaluated and improved via hyperparameter tuning.


## Date created
The project was completed on 06.12.2021.


## Usage
Point app.py to the path where your data is located and execute the script. Be sure to clean your data using the provided preprocessing functions before training the model. 


## Files used
The project uses one data file available via DataCamp: 'cc_approvals.data'

The file contains 15 columns as well as an index for each row. Here are the data types present in the file:

| column_number | data_type |
|---------------|-----------|
| 1             | object    |
| 2             | float64   |
| 3             | object    |
| 4             | object    |
| 5             | object    |
| 6             | object    |
| 7             | float64   |
| 8             | object    |
| 9             | object    |
| 10            | int64     |
| 11            | object    |
| 12            | object    |
| 13            | object    |
| 14            | int64     |
| 15            | object    |


## Credits
Sayak Paul created the project tasks for DataCamp.

The Python code in credit_card_approvals.ipynb was accepted as my solution to the project. In order to improve reusability, the code has been refactored into functions that are used by app.py.
