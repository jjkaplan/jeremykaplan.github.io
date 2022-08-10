---
layout: post
title: "Customer Churn Prediction."
subtitle: "Customer Data Analysis."

date: 2022-06-7 10:45:13 -0400
background: '/img/bank/background.png'
---


# Bank Churn Prediction


## Project Details


In this project, I will create supervised models to predict which users are more likely to churn bank. 

**The Definition of Churn Rate:**

Churn rate, in its broadest sense, is a measure of the number of individuals or items moving out of a collective group over a specific period. It is one of two primary factors that determine the steady-state level of customers a business will supportChurn rate, in its broadest sense, is a measure of the number of individuals or items moving out of a collective group over a specific period. It is one of two primary factors that determine the steady-state level of customers a business will support


## Prepare Data and Necessary Libraries 

### Install Required Library


```python
#importing necessary packages
from sklearn import model_selection
# Support functions
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# from scipy.stats import uniform
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,cross_val_score


from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import  GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Fit models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Scoring functions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score as f1
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as ex
plt.rc('figure',figsize=(18,9))
```


```python
# Set the no of columns to be displayed to 500
pd.set_option('display.max_columns', 500)

# Set the no of rows to be displayed to 300
pd.set_option('display.max_rows', 300)
```

### Load Data set


```python
#Reading files/Data

df = pd.read_csv('BankChurners.csv')
# Use 2 decimal places in output display
pd.set_option("display.precision", 2)
df.head()
```
```python
# There are some columns looks irrelevant
df = df[df.columns[:-2]]
df.head() 
```
<iframe src = "/img/bank/indext.html" height = "300px" width ="100%"></iframe>


# Data Analysing

```python
# let's look at the statistical aspects of the dataframe
df.describe()

```
<iframe src = "/img/bank/indexd.html" height = "300px" width ="100%"></iframe>


```python
# Get the no of rows and columns
print('\t Dataset has {} rows and {} columns. '.format(df.shape[0],df.shape[1]))
df.shape
```

```python
# Print the datatype of each column of the dataframe
df.info(verbose=1)
```

## Data Cleaning


```python
# Check if there any missing values
df.isna().sum()
```
```python
# Check if there is any duplicate data
df.duplicated().sum()
```
```python
# Check if there is only column with one value
df.nunique()
```
```python
#Drop row if there is any missing values

df.dropna()
```




## Data Visualizing


```python
fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['Customer_Age'],boxmean=True)
tr2=go.Histogram(x=df['Customer_Age'],name='Age Histogram')

fig.add_trace(tr1,row=1,col=1)

fig.add_trace(tr2,row=2,col=1)
fig['layout']['xaxis']['title']="Age"
fig['layout']['xaxis2']['title']="Age"
# fig['layout']['yaxis1']['title']="Count"
fig['layout']['yaxis2']['title']="Count"
fig.update_layout(height=700, width=1200, title_text="Distribution of Customer Ages", 
    yaxis_title="Count",)
fig.write_html("./file1.html")

fig.show()
```


<iframe src = "/img/bank/file1.html" height = "600px" width ="100%"></iframe>


Customer age distribution is normal

```python
fig = make_subplots(
    rows=2, cols=2,subplot_titles=('','<b>Platinum Card Holders','<b>Blue Card Holders<b>','Residuals'),
    vertical_spacing=0.09,
    specs=[[{"type": "pie","rowspan": 2}       ,{"type": "pie"}] ,
           [None                               ,{"type": "pie"}]            ,                                      
          ]
)

fig.add_trace(
    go.Pie(values=df.Gender.value_counts().values,labels=['<b>Female<b>','<b>Male<b>'],hole=0.3,pull=[0,0.3]),
    row=1, col=1
)

fig.add_trace(
    go.Pie(
        labels=['Female Platinum Card Holders','Male Platinum Card Holders'],
        values=df.query('Card_Category=="Platinum"').Gender.value_counts().values,
        pull=[0,0.05,0.5],
        hole=0.3
        
    ),
    row=1, col=2
)

fig.add_trace(
    go.Pie(
        labels=['Female Blue Card Holders','Male Blue Card Holders'],
        values=df.query('Card_Category=="Blue"').Gender.value_counts().values,
        pull=[0,0.2,0.5],
        hole=0.3
    ),
    row=2, col=2
)



fig.update_layout(
    height=800,
    showlegend=True,
    title_text="<b>Distribution Of Gender And Different Card Statuses<b>",
)

fig.show()
```


<iframe src = "/img/bank/file2.html" height = "690px" width ="100%"></iframe>


Although female are more than male in dataset, the difference is not that much large to effect the distribution.

```python
fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['Dependent_count'],name='Dependent count Box Plot',boxmean=True)
tr2=go.Histogram(x=df['Dependent_count'],name='Dependent count Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of Dependent counts (close family size)")
fig.show()
fig.write_html("./file3.html")
# fig =px.scatter(x=range(10), y=range(10))

```


<iframe src = "/img/bank/file3.html" height = "690px" width ="100%"></iframe>

```python
fig = ex.pie(df,names='Education_Level',title='Propotion Of Education Levels',hole=0.43)
fig.write_html("./file4.html")
fig.show()
```


<iframe src = "/img/bank/file4.html" height = "690px" width ="100%"></iframe>

```python
fig = ex.pie(df,names='Income_Category',title='Propotion Of Different Income Levels',hole=0.33)
fig.write_html("./file5.html")
fig.show()
```


<iframe src = "/img/bank/file5.html" height = "690px" width ="100%"></iframe>

```python
fig = ex.pie(df,names='Card_Category',title='Propotion Of Different Card Categories',hole=0.33)
fig.write_html("./file6.html")
fig.show()
```


<iframe src = "/img/bank/file6.html" height = "690px" width ="100%"></iframe>

```python
fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['Months_on_book'],name='Months on book Box Plot',boxmean=True)
tr2=go.Histogram(x=df['Months_on_book'],name='Months on book Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of months the customer is part of the bank")
fig.write_html("./file7.html")
fig.show()
```


<iframe src = "/img/bank/file7.html" height = "690px" width ="100%"></iframe>

```python
print('Kurtosis of Months on book features is : {}'.format(df['Months_on_book'].kurt()))
```

Since kurtosis is low as it clearly can be seen from graph and the result, the month as a feature is not normally distributed.

```python
fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['Total_Relationship_Count'],name='Total no. of products Box Plot',boxmean=True)
tr2=go.Histogram(x=df['Total_Relationship_Count'],name='Total no. of products Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of Total no. of products held by the customer")
fig.write_html("./file8.html")
fig.show()
```


<iframe src = "/img/bank/file8.html" height = "690px" width ="100%"></iframe>

```python
fig = ex.pie(df,names='Attrition_Flag',title='Churn Customer vs Not Churn Customer',hole=0.33)
fig.write_html("./file14.html")
fig.show()
```
<iframe src = "/img/bank/file14.html" height = "690px" width ="100%"></iframe>


The above graph says majorty of the customers are existing customers. That means the data set is not balanced thus I will upsample the dataset to balance.

```python

plt.figure(figsize = (24,170))
row_num = 50
index_pos = 1
for col in num_var:
    plt.subplot(row_num, 3, index_pos)
    sns.boxplot(y = col, x= 'Attrition_Flag', hue = 'Attrition_Flag', data = df)
    index_pos = index_pos + 1
plt.tight_layout()
plt.show()
```
```python
# Visulazitaion with heatmap to observe correlation
fig, ax = plt.subplots(figsize=(16, 12))
df_corr = df.corr(method="pearson")
# mask = np.zeros_like(np.array(df_corr))
# mask[np.triu_indices_from(mask)] = True
ax = sns.heatmap(df_corr, annot=True)
```
```python
df_copy = df.copy()
```


```python
df_copy.head()
```
# Prepare Dataset for ML trainig


```python
df_copy.columns
```
Here, dummy variable will be created for some features and also unknown columns will be dropped.

```python
# Encoding Categorical Variables
df_copy.Attrition_Flag =df_copy.Attrition_Flag.replace({'Attrited Customer':1,'Existing Customer':0})
df_copy.Gender = df_copy.Gender.replace({'F':1,'M':0})
df_copy = pd.concat([df_copy,pd.get_dummies(df_copy['Education_Level']).drop(columns=['Unknown'])],axis=1)
df_copy = pd.concat([df_copy,pd.get_dummies(df_copy['Income_Category']).drop(columns=['Unknown'])],axis=1)
df_copy = pd.concat([df_copy,pd.get_dummies(df_copy['Marital_Status']).drop(columns=['Unknown'])],axis=1)
df_copy = pd.concat([df_copy,pd.get_dummies(df_copy['Card_Category']).drop(columns=['Platinum'])],axis=1)
df_copy.drop(columns = ['Education_Level','Income_Category','Marital_Status','Card_Category','CLIENTNUM'],inplace=True)
```


```python
df_copy[df_copy.columns[:]]
```




As aformentioned , it is obvious that data is imbalanced hereI use SMOTE() method to balance data.

```python
# Here data is upsampled since the data is imbalanced
oversample = SMOTE()
X, y = oversample.fit_resample(df_copy[df_copy.columns[1:]], df_copy[df_copy.columns[0]])
usampled_df = X.assign(Churn = y)
usampled_df
```
```python
ohe_data =usampled_df[usampled_df.columns[15:-1]].copy()

usampled_df = usampled_df.drop(columns=usampled_df.columns[15:-1])

```


```python
N_COMPONENTS = 4

pca_model = PCA(n_components = N_COMPONENTS )

pc_matrix = pca_model.fit_transform(ohe_data)

evr = pca_model.explained_variance_ratio_
total_var = evr.sum() * 100
cumsum_evr = np.cumsum(evr)

trace1 = {
    "name": "individual explained variance", 
    "type": "bar", 
    'y':evr}
trace2 = {
    "name": "cumulative explained variance", 
    "type": "scatter", 
    'y':cumsum_evr}
data = [trace1, trace2]
layout = {
    "xaxis": {"title": "Principal components"}, 
    "yaxis": {"title": "Explained variance ratio"},
  }
fig = go.Figure(data=data, layout=layout)
fig.update_layout(     title='Explained Variance Using {} Dimensions'.format(N_COMPONENTS))

fig.write_html("./file9.html")
fig.show()
```


<iframe src = "/img/bank/file9.html" height = "690px" width ="100%"></iframe>

```python
usampled_df_with_pcs = pd.concat([usampled_df,pd.DataFrame(pc_matrix,columns=['PC-{}'.format(i) for i in range(0,N_COMPONENTS)])],axis=1)
```


```python
X_features = ['Total_Trans_Ct','PC-3','PC-1','PC-0','PC-2','Total_Ct_Chng_Q4_Q1','Total_Relationship_Count']

X = usampled_df_with_pcs[X_features]
y = usampled_df_with_pcs['Churn']

```


```python
# Splitted data after oversampling
X_train_smt,X_test_smt, y_train_smt, y_test_smt = train_test_split(X, y,test_size=0.2,random_state=42)

```

## Tune hyperparameters for the Best Model


Here, I will create a pipeline along with parameters for different classification model. GridSearch will be used to chose best hyperparameters among different combination of hyperparameters. 

```python
# HyperParameters for Tunning
param_grid_knn = {
# HyperParameters K Nearest Neighbors Model
'classifier__n_neighbors': np.arange(1,19,2),
"prep__num__imputer__strategy": ['mean','median']
}

param_grid_forest = {
# HyperParameters for Random Forest Model
                "classifier__n_estimators": [10, 100],
                 "classifier__max_depth":[5,8,15,25,],
                 "classifier__min_samples_leaf":[1,2,5,10,15],
                 "classifier__max_leaf_nodes": [2, 5,10]
                 }


param_grid_logReg= {
# HyperParameters for Logistic Regression Model
                "classifier__penalty": ['l2'],
                 "classifier__C": np.logspace(0, 4, 10)
                 }
param_grid_svc ={
# HyperParameters for Support Vector Machne Model
                  'classifier__gamma': [0.01],
                  'classifier__kernel': ['linear', 'poly', 'rbf'],
                 'classifier__C': [100]
                 }
```

```python
# Here created a function for pipline
def pipline_model(model):

    numeric_feature = ['Total_Trans_Ct','PC-3','PC-1','PC-0','PC-2','Total_Ct_Chng_Q4_Q1','Total_Relationship_Count']
    numeric_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy= 'median')),
    ('scaler',StandardScaler())                                      
    ])
    preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_feature)    
    ])
    clf = Pipeline(steps=[
    ('prep', preprocessor),
    ('classifier',model),
    ])
    return clf
```

```python
# Creating a variable for pipeline_model function for each model.
clf_svc = pipline_model(SVC(probability=True))
clf_random_forest = pipline_model(RandomForestClassifier())
clf_knn = pipline_model(KNeighborsClassifier())
clf_logisticreg = pipline_model(LogisticRegression( max_iter=1000)) 

```

```python
# Creating a variable for Gridsearch for each model
grid_search_forest = GridSearchCV(clf_random_forest,param_grid,cv=10,scoring='neg_mean_squared_error')
grid_search_knn = GridSearchCV(clf_knn,param_grid,cv=10)
grid_search_svc = GridSearchCV(clf_svc,param_grid,refit=True,verbose=2) 
grid_search_logReg = GridSearchCV(clf_logisticreg,param_grid,cv=10)
```

```python
## Created a function run all four models with their pipelnes to compare their results.

knn=    1
forest= 2
svc=    3
logisticreg= 4

def model_score(GSCV):
    
    
    if GSCV == 1:
    
        grid_search_knn = GridSearchCV(clf_knn,param_grid,cv=10)
        grid_search_knn.fit(X_train_smt,y_train_smt)
        pred=grid_search_knn.predict(X_test_smt)
        print("Confusion matrix of knn:\n ",confusion_matrix(y_test_smt,pred))
        print("\n\n")
        print(classification_report(y_test_smt,pred))
        score= grid_search_knn.score(X_test_smt,y_test_smt)  
        print('Score of knneighbors is ',score)
        cros_val_score =  cross_val_score(clf_knn,X_train_smt, y_train_smt).mean()
        print("Cross validation score of knneighbors is ", cros_val_score)

    
    
    elif GSCV == 2:
        grid_search_forest = GridSearchCV(clf_random_forest,param_grid,cv=10)
        grid_search_forest.fit(X_train_smt,y_train_smt)
        
        pred=grid_search_forest.predict(X_test_smt)
        print("Confusion matrix of random forest:\n ",confusion_matrix(y_test_smt,pred))
        print("\n\n")
        print(classification_report(y_test_smt,pred))
        
        score= grid_search_forest.score(X_test_smt,y_test_smt)
        print('Score of random forest is ',score)
        cros_val_score =  cross_val_score(clf_random_forest,X_train_smt, y_train_smt).mean()
        print("Cross validation score of random forest is ", cros_val_score)

    

    elif GSCV == 3:
        ggrid_search_svc = GridSearchCV(clf_svc,param_grid,refit=True,verbose=2) 
        ggrid_search_svc.fit(X_train_smt,y_train_smt)
        
        pred=ggrid_search_svc.predict(X_test_smt)
        print("Confusion matrix of SVC:\n ",confusion_matrix(y_test_smt,pred))
        print("\n\n")
        print(classification_report(y_test_smt,pred))
        
        score=  ggrid_search_svc.score(X_test_smt,y_test_smt)  
        print('The score of svc is ',score)
        cros_val_score =  cross_val_score(clf_svc,X_train_smt, y_train_smt).mean()
        print("the cross validation score of SVC is ", cros_val_score)

    elif GSCV == 4:
        grid_search_logReg = GridSearchCV(clf_logteg,param_grid,cv=10)
        grid_search_logReg.fit(X_train_smt,y_train_smt)
        
        pred=grid_search_logReg.predict(X_test_smt)
        print("Confusion matrix of logisticReg :\n ",confusion_matrix(y_test_smt,pred))
        print("\n\n")
        print(classification_report(y_test_smt,pred))
        
        score=  grid_search_logReg.score(X_test_smt,y_test_smt)
        print('The score of rlogistic regression is ',score)
        cros_val_score =  cross_val_score(clf_logteg,X_train_smt, y_train_smt).mean()
        print("the cross validation score of logistic regression is ", cros_val_score)
 

```

```python
# Here I used VotingClassifier to select best model among four models.
from sklearn.ensemble import VotingClassifier

model = VotingClassifier(estimators=[('lr', grid_search_logReg), ('knn',grid_search_knn),('forest',grid_search_forest),('svc', grid_search_svc)], voting='soft')
model.fit(X_train_smt,y_train_smt)

```

```python
print(model.score(X_test_smt,y_test_smt))

```

Random Forest Model gives the highest result  among the models, however Votingclasiffier select Sopport Vector machine model although it is slightly smaller than Random Forest Model. Since SVM is selected, I will save this model as a reference.


##  Model Selection


```python
grid_search_svc = GridSearchCV(clf_svc,param_grid_svc,refit=True,verbose=2) 
best_model = grid_search_svc.fit(X_train_smt,y_train_smt)
```

```python
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
# saving the model to local disk
# filename = 'finalized_model.sav'
# joblib.dump(best_model, filename)

 
# load the model from local disk to use it
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test_smt, y_test_smt)
print(result)
print(loaded_model.best_estimator_)
print(loaded_model.best_score_)
predict = loaded_model.predict(X_test_smt)
print("accuracy ", accuracy_score(y_test_smt,predict))
print("f1score ", f1_score(y_test_smt,predict))
```

```python
# 
rf_f1_cross_val_scores = cross_val_score(clf_random_forest,X_train_smt,y_train_smt,cv=5,scoring='f1')
knn_f1_cross_val_scores=cross_val_score(clf_knn,X_train_smt,y_train_smt,cv=5,scoring='f1')
svm_f1_cross_val_scores=cross_val_score(clf_svc,X_train_smt,y_train_smt,cv=5,scoring='f1')
lreg_f1_cross_val_scores=cross_val_score(clf_logisticreg,X_train_smt,y_train_smt,cv=5,scoring='f1')
```


```python
fig = make_subplots(rows=4, cols=1,shared_xaxes=True,subplot_titles=('Random Forest Cross Val Scores',
                                                                     'KNN Cross Val Scores',
                                                                    'SVM Cross Val Scores', 'Logistic Regression'))

fig.add_trace(
    go.Scatter(x=list(range(0,len(rf_f1_cross_val_scores))),y=rf_f1_cross_val_scores,name='Random Forest'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=list(range(0,len(knn_f1_cross_val_scores))),y=knn_f1_cross_val_scores,name='KNN'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=list(range(0,len(svm_f1_cross_val_scores))),y=svm_f1_cross_val_scores,name='SVM'),
    row=3, col=1
)
fig.add_trace(
    go.Scatter(x=list(range(0,len(lreg_f1_cross_val_scores))),y=lreg_f1_cross_val_scores,name='Logistic Regression'),
    row=4, col=1
)

fig.update_layout(height=700, width=900, title_text="Different Model 5 Fold Cross Validation")
fig.update_yaxes(title_text="F1 Score")
fig.update_xaxes(title_text="Fold #")
# fig.write_html("./file15.html")

fig.show()
```


<iframe src = "/img/bank/file15.html" height = "690px" width ="100%"></iframe>

```python
clf_random_forest.fit(X_train_smt,y_train_smt)
rf_prediction = clf_random_forest.predict(X_test_smt)

clf_knn.fit(X_train_smt,y_train_smt)
knn_prediction = clf_knn.predict(X_test_smt)

clf_svc.fit(X_train_smt,y_train_smt)
svm_prediction = clf_svc.predict(X_test_smt)

clf_logisticreg.fit(X_train_smt,y_train_smt)
lreg_prediction = clf_logisticreg.predict(X_test_smt)

```


```python
fig = go.Figure(data=[go.Table(header=dict(values=['<b>Model<b>', '<b>F1 Score On Test Data<b>'],
                                           line_color='darkslategray',
    fill_color='whitesmoke',
    align=['center','center'],
    font=dict(color='black', size=18),
    height=40),
                               
                 cells=dict(values=[['<b>Random Forest<b>', '<b>KNN<b>','<b>SVM<b>','<b>LogisticReg<b>'], [np.round(f1(rf_prediction,y_test_smt),2), 
                                                                          np.round(f1(knn_prediction,y_test_smt),2),
                                                                          np.round(f1(svm_prediction,y_test_smt),2),
                                                                         np.round(f1(lreg_prediction,y_test_smt),2)]]))
                     ])

fig.update_layout(title='Model Results On Test Data')
fig.write_html("./file10.html")
fig.show()
```


<iframe src = "/img/bank/file10.html" height = "690px" width ="100%"></iframe>


```python
ohe_data =df_copy[df_copy.columns[16:]].copy()
pc_matrix = pca_model.fit_transform(ohe_data)
original_df_with_pcs = pd.concat([df_copy,pd.DataFrame(pc_matrix,columns=['PC-{}'.format(i) for i in range(0,N_COMPONENTS)])],axis=1)

unsampled_data_prediction_RF = clf_random_forest.predict(original_df_with_pcs[X_features])
unsampled_data_prediction_knn = clf_knn.predict(original_df_with_pcs[X_features])
unsampled_data_prediction_SVM = clf_svc.predict(original_df_with_pcs[X_features])
unsampled_data_prediction_LR = clf_logisticreg.predict(original_df_with_pcs[X_features])
```


```python
fig = go.Figure(data=[go.Table(header=dict(values=['<b>Model<b>', '<b>F1 Score On Test Data<b>'],
                                           line_color='darkslategray',
    fill_color='whitesmoke',
    align=['center','center'],
    font=dict(color='black', size=18),
    height=40),
                               
                 cells=dict(values=[['<b>Random Forest<b>', '<b>KNN<b>','<b>SVM<b>','<b>LogisticReg<b>'], [np.round(f1(unsampled_data_prediction_RF,original_df_with_pcs['Attrition_Flag']),2), 
                                                                          np.round(f1(unsampled_data_prediction_knn,original_df_with_pcs['Attrition_Flag']),2),
                                                                          np.round(f1(unsampled_data_prediction_SVM,original_df_with_pcs['Attrition_Flag']),2),
                                                                         np.round(f1(unsampled_data_prediction_LR,original_df_with_pcs['Attrition_Flag']),2)]]))
                     ])

fig.update_layout(title='F1 Scores for Models Before Upsampling Data')
fig.write_html("./file11.html")
fig.show()
```


<iframe src = "/img/bank/file11.html" height = "690px" width ="100%"></iframe>


```python
# Creating f1 scores of models
score_list = [np.round(f1(unsampled_data_prediction_RF,original_df_with_pcs['Attrition_Flag']),2),
              np.round(f1(unsampled_data_prediction_knn,original_df_with_pcs['Attrition_Flag']),2),
             np.round(f1(unsampled_data_prediction_SVM,original_df_with_pcs['Attrition_Flag']),2),
             np.round(f1(unsampled_data_prediction_LR,original_df_with_pcs['Attrition_Flag']),2)]
```

```python
# Creating dataFrame for models score
Model_Scores = pd.DataFrame([score_list],
                      columns=[['Random Forest', 'KNN','SVM','LogisticReg']])

Model_Scores.head(5)
```

From above table it is clear that without balancing data, model perfotmances are quite weak.

```python
import plotly.figure_factory as ff
z=confusion_matrix(unsampled_data_prediction_SVM,original_df_with_pcs['Attrition_Flag'])
fig = ff.create_annotated_heatmap(z, x=['Not Churn','Churn'], y=['Predicted Not Churn','Predicted Churn'], colorscale='Viridis',xgap=3,ygap=3)
fig['data'][0]['showscale'] = True
fig.update_layout(title='Prediction On Original Data With Support Vector Machine Model Confusion Matrix')
fig.write_html("./file12.html")
fig.show()

```


<iframe src = "/img/bank/file12.html" height = "690px" width ="100%"></iframe>


```python
import scikitplot as skplt
unsampled_data_prediction_SVC = clf_svc.predict_proba(original_df_with_pcs[X_features])
skplt.metrics.plot_precision_recall(original_df_with_pcs['Attrition_Flag'], unsampled_data_prediction_SVC)
plt.legend(prop={'size': 20})
plt.title("Precision-Recall Curve for Support Vector Machine Model")

fig.show()
```
<iframe src = "/img/bank/curve.png" height = "690px" width ="100%"></iframe>



# Conlusion


In this project Bank customer churn data analyzed. There is huge difference between the balanced and imbalnced data results. 
Here the importnce of resampling SMOTE() method's importance comes into play. 

Another point is that despite obtaining higher result from random forest model, VotingClassfier, which select best model among introduced model, selected support vector machine model


