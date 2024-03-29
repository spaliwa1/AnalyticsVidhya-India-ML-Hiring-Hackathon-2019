Analytics Vidhya - India ML Hiring Hackathon 2019
My solution to AV - ML Hackathon 2019.

Loan Delinquency Prediction

Loan default prediction is one of the most critical and crucial problem faced by financial institutions and organizations as it has a noteworthy effect on the profitability of these institutions. In recent years, there is a tremendous increase in the volume of non – performing loans which results in a jeopardizing effect on the growth of these institutions. Therefore, to maintain a healthy portfolio, the banks put stringent monitoring and evaluation measures in place to ensure timely repayment of loans by borrowers. Despite these measures, a major proportion of loans become delinquent. Delinquency occurs when a borrower misses a payment against his/her loan.
Given the information like mortgage details, borrowers related details and payment details, our objective is to identify the delinquency status of loans for the next month given the delinquency status for the previous 12 months (in number of months)

Data Description
train.zip
train.zip contains train.csv. train.csv contains the training data with details on loan as described in the last section

Data Dictionary:

•	loan_id Unique loan ID

•	source Loan origination channel

•	financial_institution Name of the bank

•	interest_rate Loan interest rate

•	unpaid_principal_bal Loan unpaid principal balance

•	loan_term Loan term (in days)

•	origination_date Loan origination date (YYYY-MM-DD)

•	first_payment_date First instalment payment date

•	loan_to_value Loan to value ratio

•	number_of_borrowers Number of borrowers

•	debt_to_income_ratio Debt-to-income ratio

•	borrower_credit_score Borrower credit score

•	loan_purpose Loan purpose

•	insurance_percent Loan Amount percent covered by insurance

•	co-borrower_credit_score Co-borrower credit score

•	insurance_type 0 - Premium paid by borrower, 1 - Premium paid by Lender

•	m1 to m12 Month-wise loan performance (deliquency in months)

•	m13 target, loan deliquency status (0 = non deliquent, 1 = deliquent)

test.zip

test.zip contains test.csv which has details of all loans for which the participants are to submit the delinquency status - 0/1 (not probability)
sample_submission.zip

sample_submission.zip contains the submission format for the predictions against the test set. A single csv needs to be submitted as a solution.

Evaluation Metric

Submissions are evaluated on F1-Score between the predicted class and the observed target.

Approach

The very step was to explore and visualize the data and perform basis check on it to get familiar with it. Target variable was highly imbalanced so I got an idea that for this model I have to try different sampling techniques and check which one is best for this model. Checked for missing value, luckily there were none in this dataset. Checked for correlation if any among columns. Created dummies for categorical variables. Created a new feature totaldays by subtracting origination date and first payment date. I have tried to use PCA and SMOTE(in combination) for this particular dataset with different ML algos but result was not that good. Finally I have decided to go without PCA as  dimensionality was not a concern for this data set. I have used different models but Adaboost classifier has given me the best result. For parameter tuning ,I have used grid search. 

Packages Used

from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV

How to use

run the Python3 India ML Solution.py file.	





