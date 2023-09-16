#!/usr/bin/env python
# coding: utf-8

# Importing the Dependencies

# In[100]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# In[101]:


# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv("creditcard.csv")


# In[102]:


# first 5 rows of the dataset
credit_card_data.head()


# In[103]:


credit_card_data.tail()


# In[104]:


# dataset informations
credit_card_data.info()


# In[105]:


# checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[106]:


# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()


# This Dataset is highly unblanced

# 0 --> Normal Transaction
# 
# 1 --> fraudulent transaction

# In[107]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[108]:


print(legit.shape)
print(fraud.shape)


# In[109]:


# statistical measures of the data
legit.Amount.describe()


# In[110]:


fraud.Amount.describe()


# In[111]:


# compare the values for both transactions
credit_card_data.groupby('Class').mean()


# Under-Sampling

# Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions

# Number of Fraudulent Transactions --> 492

# In[112]:


legit_sample = legit.sample(n=492)


# Concatenating two DataFrames

# In[113]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[114]:


new_dataset.head()


# In[115]:


new_dataset.tail()


# In[116]:


new_dataset['Class'].value_counts()


# In[117]:


new_dataset.groupby('Class').mean()


# Splitting the data into Features & Targets

# In[118]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[119]:


print(X)


# In[120]:


print(Y)


# Split the data into Training data & Testing Data

# In[121]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[122]:


print(X.shape, X_train.shape, X_test.shape)


# # Model Training

# # Logistic Regression

# In[123]:


from sklearn.linear_model import LogisticRegression


# In[124]:


model = LogisticRegression()


# In[125]:


# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score

# In[186]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[187]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[190]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Logistic Regression - Test Data Accuracy:', logistic_accuracy)


# In[194]:


# Calculate F1 score on test data
logistic_f1 = f1_score(X_test_prediction, Y_test)
print('Logistic Regression - F1 Score:', logistic_f1)


# # KNN

# In[131]:


from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


# In[132]:


n = 7
KNN = KNeighborsClassifier(n_neighbors = n)


# In[133]:


KNN.fit(X_train, Y_train)


# Model evaluation and Accuracy Score

# In[163]:


# accuracy on training data
X_train_prediction = KNN.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[164]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[199]:


## accuracy on test data
X_test_prediction = KNN.predict(X_test)
knn_accuracy = accuracy_score(X_test_prediction, Y_test)
print('K-Nearest Neighbors (KNN) - Test Data Accuracy:', knn_accuracy)


# In[200]:


# Calculate F1 score on test data
knn_f1 = f1_score(X_test_prediction, Y_test)
print('K-Nearest Neighbors (KNN) - F1 Score:', knn_f1)


# # Decision Tree 

# In[139]:


from sklearn.tree import DecisionTreeClassifier


# In[140]:


DT = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')


# In[141]:


DT.fit(X_train, Y_train)


# Model evaluation and Accuracy Score

# In[142]:


# accuracy on training data
X_train_prediction = DT.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[143]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[197]:


## accuracy on test data
X_test_prediction = DT.predict(X_test)
decision_tree_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Decision Tree - Test Data Accuracy:',  decision_tree_accuracy)


# In[198]:


# Calculate F1 score on test data
decision_tree_f1 = f1_score(X_test_prediction, Y_test)
print('Decision Tree - F1 Score:',  decision_tree_f1)


# # RANDOMFOREST
# 

# In[147]:


from sklearn.ensemble import RandomForestClassifier


# In[148]:


rf = RandomForestClassifier(max_depth = 4)


# In[149]:


rf.fit(X_train, Y_train)


# model evaluation and accuracy score

# In[150]:


# accuracy on training data
X_train_prediction = rf.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[151]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[202]:


## accuracy on test data
X_test_prediction = rf.predict(X_test)
random_forest_accuracy = accuracy_score(X_test_prediction, Y_test)
# Print the results
print('Random Forest - Test Data Accuracy:', random_forest_accuracy)


# In[203]:


# Calculate F1 score on test data
random_forest_f1 = f1_score(X_test_prediction, Y_test)
print('Random Forest - F1 Score:', random_forest_f1)


# In[204]:


results = pd.DataFrame({
    'Model': ['Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree', 'Random Forest'],
    'Test Data Accuracy': [logistic_accuracy, knn_accuracy, decision_tree_accuracy, random_forest_accuracy],
    'F1 Score': [logistic_f1, knn_f1, decision_tree_f1, random_forest_f1]
})
# Print and compare the results
print(results)


# In[207]:


import matplotlib.pyplot as plt

# Model names
models = ["Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest"]

# Test data accuracy values
accuracy_scores = [logistic_accuracy, knn_accuracy, decision_tree_accuracy, random_forest_accuracy]

# F1 score values
f1_scores = [logistic_f1, knn_f1, decision_tree_f1, random_forest_f1]

# Set the width of the bars
bar_width = 0.35

# Position of bars on x-axis
x = np.arange(len(models))

# Create grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Plot test data accuracy
ax.bar(x - bar_width/2, accuracy_scores, bar_width, label='Test Data Accuracy', color='skyblue')
# Plot F1 scores
ax.bar(x + bar_width/2, f1_scores, bar_width, label='F1 Score', color='lightcoral')

# Set labels, title, and legend
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right")
ax.legend()

plt.tight_layout()
plt.show()


# In conclusion :
# 
# 1. **Logistic Regression:** This model exhibits relatively high accuracy but a lower F1 score, suggesting a trade-off between overall accuracy and precision-recall balance. If minimizing both false positives and false negatives is not a top priority, it remains a viable choice.
# 
# 2. **K-Nearest Neighbors (KNN):** In comparison to other models, KNN demonstrates lower accuracy, with a F1 score similar to Logistic Regression. This indicates that KNN may not be the optimal solution for this dataset, offering lower accuracy without significant improvements in performance.
# 
# 3. **Decision Tree:** The Decision Tree model performs well in terms of both accuracy and F1 score, boasting a high level of accuracy and an F1 score near unity. This reflects a strong balance between precision and recall, making it a promising option.
# 
# 4. **Random Forest:** Random Forest surpasses the Decision Tree model in accuracy and F1 score. It achieves a favorable blend of precision and recall. Random Forest emerges as another appealing choice for this dataset.
# 
# In summary, both the Decision Tree and Random Forest models demonstrate strong performance regarding accuracy and F1 score, positioning them as viable options for the credit card fraud detection task. While Logistic Regression is accurate, it may not achieve the same level of precision as other approaches. K-Nearest Neighbors falls short in terms of accuracy and may be less desirable in this context. Ultimately, the choice between the Decision Tree and Random Forest models may hinge on factors such as interpretability, computational resources, and the need for ensemble learning.

# In[ ]:




