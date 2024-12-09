import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
kidney_df=pd.read_csv("/Users/rameezmohammed/desktop/kidney_disease.csv")
print(kidney_df.sample(10))
print(kidney_df.shape)
print(kidney_df.info())
kidney_df_temp=kidney_df.drop(['id'],axis=1)
kidney_df_temp.columns=['age','blood_pressure','specific_gravity','albumin','sugar','red_blood_cells','pus_celss','puss_cells_clumps','bacteria','blood_glucose_random','blood_urea','serum_creatinine',
                   'sodium','potassium','haemoglobin','packed_cell_volume','white_blood_cell_count','red_blood_cell_count','hypertension','diabetes_mellitus','coronary_artery_disease','appetite',
                   'peda_edema','anemia','class']
print(kidney_df_temp.head())   
print(kidney_df_temp['albumin'].unique())
kidney_df_temp[['specific_gravity','albumin','sugar']]= kidney_df_temp[['specific_gravity','albumin','sugar']].astype(object)    
print(kidney_df_temp['packed_cell_volume'].unique())
print(kidney_df_temp['white_blood_cell_count'].unique())
print(kidney_df_temp['red_blood_cell_count'].unique())
kidney_df_temp['packed_cell_volume']=pd.to_numeric(kidney_df_temp['packed_cell_volume'],errors='coerce')
kidney_df_temp['white_blood_cell_count']=pd.to_numeric(kidney_df_temp['white_blood_cell_count'],errors='coerce')
kidney_df_temp['red_blood_cell_count']=pd.to_numeric(kidney_df_temp['red_blood_cell_count'],errors='coerce')
categorical=kidney_df_temp.select_dtypes(include=['object'])
print(f'categorical columns: {categorical.columns}')
num_col=[col for col in kidney_df_temp.columns if col not in categorical.columns]
print(f'numerical columns: {num_col}')
for col in categorical.columns:
    print(f'{col} has {kidney_df_temp[col].unique()} values\n')
kidney_df_temp['diabetes_mellitus']=kidney_df_temp['diabetes_mellitus'].replace({' yes':'yes','\tno':'no','\tyes':'yes'})
kidney_df_temp['coronary_artery_disease']=kidney_df_temp['coronary_artery_disease'].replace({'\tno':'no'})
kidney_df_temp['class']=kidney_df_temp['class'].replace({'ckd\t':'ckd'})
kidney_df_temp['class']=kidney_df_temp['class'].map({'ckd':0,'notckd':1})
kidney_df_temp['class']=pd.to_numeric(kidney_df_temp['class'],errors='coerce')
print(kidney_df_temp['class'].dtype)
kidney_df_temp1=kidney_df_temp.select_dtypes(include=['number'])
plt.figure(figsize=(15,8))
sns.heatmap(kidney_df_temp1.corr(),center=0,annot=True)
plt.title('correlation matrix')
plt.show()
def mode_impuration(feature):
    if not kidney_df_temp[feature].mode().empty:
        mode1=kidney_df_temp[feature].mode()[0]
        kidney_df_temp[feature]=kidney_df_temp[feature].fillna(mode1)
def random_sample_imputation(feature):
    non_null_values=kidney_df_temp[feature].dropna().values
    random_samples=np.random.choice(non_null_values,kidney_df_temp[feature].isnull().sum(),replace=True)
    kidney_df_temp.loc[kidney_df_temp[feature].isnull(),feature]=random_samples
for col in num_col:
    random_sample_imputation(col)
print(kidney_df_temp[num_col].isnull().sum().sort_values(ascending=False))
random_sample_imputation('red_blood_cells')
random_sample_imputation('pus_celss')
for col in categorical.columns:
    mode_impuration(col)
print(kidney_df_temp[categorical.columns].isnull().sum().sort_values(ascending=False))
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in categorical.columns[3:]:
    kidney_df_temp[col]=le.fit_transform(kidney_df_temp[col])
print(kidney_df_temp.head())
x=kidney_df_temp.drop('class',axis=1)
y=kidney_df_temp['class']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train).ravel()
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
print(type(x_train))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_train_pred=knn.predict(x_train)
y_test_predict=knn.predict(x_test)
knn_acc=accuracy_score(y_test,y_test_predict )
print(f'knn training accuracy: {accuracy_score(y_train, y_train_pred)}')
print(f'knn test accuracy: {knn_acc}')
print(f'knn confusion matrix: {confusion_matrix(y_test,y_test_predict)}')
print(f'knn classification report: \n {classification_report(y_test,y_test_predict)}')    
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_train_pred=dtc.predict(x_train)
y_test_predict=dtc.predict(x_test)
dtc_acc=accuracy_score(y_test,y_test_predict)
print(f'decision tree training accuracy: {accuracy_score(y_train, y_train_pred)}')
print(f'decision tree test accuracy: {dtc_acc}')
print(f'decision tree confusion matrix: {confusion_matrix(y_test,y_test_predict)}')
print(f'decision tree classification report: \n {classification_report(y_test, y_test_predict)}')  
from sklearn.model_selection import GridSearchCV
parameters={'criterion':['gini','entropy'],
            'max_depth':[3,5,7,10],
            'splitter':['best','random'],
            'min_samples_leaf':[1,2,3,5,7],
            'min_samples_split':[1,2,3,5,7],
            'max_features':['auto','sqrt','log2']}
grid_dtc=GridSearchCV(estimator=dtc,param_grid=parameters,cv=5,n_jobs=-1,verbose=1)
grid_dtc.fit(x_train,y_train)
print(grid_dtc.best_params_)
print(grid_dtc.best_score_)
dtc=grid_dtc.best_estimator_
dtc.fit(x_train,y_train)
y_train_pred=dtc.predict(x_train)
y_test_predict=dtc.predict(x_test)
print(f'decision tree training accuracy: {accuracy_score(y_train, y_train_pred)}')
print(f'decision tree test accuracy: {accuracy_score(y_test,y_test_predict)}')
print(f'decision tree confusion matrix: {confusion_matrix(y_test,y_test_predict)}')
print(f'decision tree classification report: \n {classification_report(y_test,y_test_predict)}')  
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
rfc=RandomForestClassifier(criterion='entropy',max_depth=11,max_features='sqrt',min_samples_leaf=2,min_samples_split=3,n_estimators=130)
rfc.fit(x_train,y_train)
y_train_pred=rfc.predict(x_train)
y_test_predict=rfc.predict(x_test)
rfc_acc=accuracy_score(y_test,y_test_predict )
print(f'random forest training accuracy: {accuracy_score(y_train, y_train_pred)}')
print(f'random forest test accuracy: {rfc_acc}')
print(f'random forest confusion matrix: {confusion_matrix(y_test,y_test_predict) }')
print(f'random forest classification report: \n {classification_report(y_test,y_test_predict)}')  
etc=ExtraTreesClassifier()
etc.fit(x_train,y_train)
y_train_pred=etc.predict(x_train)
y_test_predict=etc.predict(x_test)
etc_acc=accuracy_score(y_test,y_test_predict)
print(f'extra tree training accuracy: {accuracy_score(y_train, y_train_pred)}')
print(f'extra tree test accuracy: {etc_acc}')
print(f'extra tree confusion matrix: {confusion_matrix(y_test,y_test_predict)}')
print(f'extra tree classification report: \n {classification_report(y_test,y_test_predict)}')  
models=pd.DataFrame({'model':['knn classifier','decision tree classifier','random forest classifier','extra tree classifier'],
                    'score':[knn_acc,dtc_acc,rfc_acc,etc_acc]})
print(models.sort_values(by='score',ascending=False))