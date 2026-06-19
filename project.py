# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Train-test split
from sklearn.model_selection import train_test_split

# Model
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
df=pd.read_csv("c.csv") #import the data file

print(df.head()) # first ki rows dekhna just to check
print(df.shape)
print(df.columns) #gives the column names of all

print(df.isnull().sum()) #sari jgah jha null h vo output ayega

print(df["target"].value_counts()) #yh btayega counts of diseass and not disesed
#this is used to check ki humara datatset balanced h ya imbaalanced
#like humare case m balanced h
X=df.drop("target",axis=1) #full dataset except target
#axis=1 means column axis=0 means rows
y=df["target"] #only target
print(X.shape)
print(y.shape)
#shape func gives rpws and columns
X_train,X_test,y_train,y_test=train_test_split(
    X, #tarining aur testing k lie split kr rhe h
    y,
    test_size=0.2,  #20-80 ratio
    random_state=42  #set seed
)
print(X_train.shape)
print(y_train.shape)

model=RandomForestClassifier(
    random_state=42  #a set of many decision trees
)
model.fit(X_train,y_train)  #train krna model ko

y_pred= model.predict(X_test) #predictions
print(y_pred[:10])

print(accuracy_score(y_test,y_pred))

conf=confusion_matrix(y_test,y_pred)
print(conf)

print(classification_report(y_test,y_pred))

y_prob = model.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC Score:", auc)

plt.figure(figsize=(6,4))
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance.sort_values().plot(
    kind='barh',
    figsize=(8,5)
)

plt.title("Feature Importance")
plt.show()
