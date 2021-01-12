import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix,recall_score,f1_score,classification_report
import pandas as pd  # import libary ต่างๆที่จะใช้
from sklearn.linear_model import LogisticRegression

#Data Preparation
social_data = pd.read_csv("C:\\Users\\Admin\\Desktop\\PortGT03\\SocialNetworkAds.csv")
df_social = pd.get_dummies(social_data,columns=['Gender']) # One hot encoding for gender
print(df_social.head()) # Show data
print(df_social.describe()) #Describe all value
print('check number =',df_social.nunique()) # number of unique variable
print('check null =',df_social.isnull().sum()) # check Null
data_x = df_social.drop(['User ID','Purchased'],axis=1)
data_y = df_social['Purchased']

#Heatmap and count and pairplot

figure1 = plt.figure()
sns.heatmap(df_social.corr() , square=True, fmt='.1f', annot=True, cmap='Reds') # heat map for analysis
figure2 = plt.figure()
sns.countplot(df_social['Purchased'])
figure3 = plt.figure()
sns.countplot(df_social['Age'])
sns.pairplot(df_social) # เทียบทุกกรณี

#Train and Logistic Regression

X_train, X_test, y_train, y_test = train_test_split(data_x,data_y, random_state=40,train_size= 0.5) # test 50% train 50%
lg = LogisticRegression()
lg.fit(X_train,y_train)
y_pred = lg.predict(X_test)

#Model Evaluation

print("Score = ",lg.score(X_test,y_test))
print("Model Score : ",accuracy_score(y_test,y_pred))
print("Precision Score :",precision_score(y_test,y_pred))
print("F1 Score : ", f1_score(y_test,y_pred))
print("Recall ",recall_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test , y_pred))

#plot confusion matrix

figure4 = plt.figure()

axis=confusion_matrix(y_test,y_pred)

sns.heatmap(axis,annot=True,cmap='Reds')

plt.title('Confusion Matrix')




plt.show()

