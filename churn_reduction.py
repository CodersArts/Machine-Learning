import pandas as pd
import numpy as np
from IPython import get_ipython
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# creating a dict file
yesno = {'yes': 1, 'no': 0}

df_train = pd.read_csv("Train_data.csv")
df_test = pd.read_csv("Test_data.csv")

# translating yes/no value with 1/0 in data frame
df_train["international plan"] = [yesno[item.strip()] for item in df_train["international plan"]]
df_train["voice mail plan"] = [yesno[item.strip()] for item in df_train["voice mail plan"]]
df_test["international plan"] = [yesno[item.strip()] for item in df_test["international plan"]]
df_test["voice mail plan"] = [yesno[item.strip()] for item in df_test["voice mail plan"]]

# display(df.head(5))
print("Number of rows: ", df_train.shape[0])
counts = df_train.describe().iloc[0]
display(
    pd.DataFrame(
        counts.tolist(),
        columns=["Count of values"],
        index=counts.index.values
    ).transpose()
)
# Drop the columns that we have decided won't be used in prediction
df_train = df_train.drop(["state", "area code", "phone number"], axis=1)
features = df_train.drop(["Churn"], axis=1).columns

# Set up our RandomForestClassifier instance and fit to data
clf = RandomForestClassifier(n_estimators=30)
clf.fit(df_train[features], df_train["Churn"])

# Make predictions
predictions = clf.predict(df_test[features])
probs = clf.predict_proba(df_test[features])
print(probs)
display(predictions)






