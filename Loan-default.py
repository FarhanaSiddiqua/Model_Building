import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.feature_selection import f_regression , SelectKBest , f_classif, mutual_info_classif
pd.options.display.max_columns = 775
pd.options.display.max_rows = 775
from sklearn.metrics import r2_score , classification_report, ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split , cross_val_score,StratifiedKFold,RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel,RFECV, RFE
from sklearn.linear_model import  LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier  
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import warnings
from sklearn.ensemble import GradientBoostingClassifier
warnings.simplefilter("ignore")
train=pd.read_csv('/Users/rameezmohammed/downloads/train_v2.csv',index_col='id')
test=pd.read_csv('/Users/rameezmohammed/downloads/test_v2.csv',index_col='id')
print(train.shape) #(105471,770)
print(train.head())
print(train.info())
train.loss.loc[train.loss!=0]=1
print(train.loss.value_counts())
train.drop_duplicates()
print(train.info())
for col in train.select_dtypes('number'):
    train[col]=pd.to_numeric(train[col],downcast='integer')
    if train[col].dtype=='float':
        train[col]=pd.to_numeric(train[col],downcast='float')
print(train.info()) #here we are saving the memory by downcasting the datatypes
num_cols=[col for col in train.columns if train[col].dtype!='object']
train=train[num_cols]
print(train.shape)
train=train.dropna(axis=0,subset=('loss'))
x=train.drop('loss',axis=1)
y=train.loss
print(x.shape)
scaler=StandardScaler()
for col in x.columns:
    x[col]=x[col].fillna(x[col].mean())
    x[col]=scaler.fit_transform(x[[col]])
scaled_data=pd.DataFrame(x)
pca=PCA(n_components=0.95)
pca.fit(scaled_data)
x_scaled=pca.transform(scaled_data)
print(x_scaled.shape)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(10,10))
axis=fig.add_subplot(111,projection='3d')
axis.scatter(x_scaled[:,0],x_scaled[:,1],x_scaled[:,2],c=y,cmap='RdBu')
axis.set_xlabel('pc1',fontsize=10)
axis.set_ylabel('pc2',fontsize=10)
axis.set_zlabel('pc3',fontsize=10)
plt.legend([0,1])
print(plt.show())
fig=plt.figure(figsize=(10,10))
axis=fig.add_subplot(111,projection='3d')
axis.scatter(x_scaled[:,5],x_scaled[:,6],x_scaled[:,7],c=y,cmap='RdBu')
axis.set_xlabel('pc1',fontsize=10)
axis.set_ylabel('pc2',fontsize=10)
axis.set_zlabel('pc3',fontsize=10)
plt.legend([0,1])
print(plt.show())
fig=plt.figure(figsize=(10,10))
axis=fig.add_subplot(111,projection='3d')
axis.scatter(x_scaled[:,100],x_scaled[:,101],x_scaled[:,102],c=y,cmap='RdBu')
axis.set_xlabel('pc1',fontsize=10)
axis.set_ylabel('pc2',fontsize=10)
axis.set_zlabel('pc3',fontsize=10)
plt.legend([0,1])
print(plt.show()) #all pcs are overlapped,there is no separation between the pcs
num_cols=[col for col in train.columns if train[col].dtype!='object']
train=train[num_cols]
print(train.shape)
train=train.dropna(axis=0,subset=('loss'))
f1=train.drop('loss',axis=1)
t1=train.loss
scaler=StandardScaler()
for col in f1.columns:
    f1[col]=f1[col].fillna(f1[col].mean())
    f1[col]=scaler.fit_transform(f1[[col]])
x_train,x_test,y_train,y_test=train_test_split(f1,t1,test_size=0.1,random_state=30)
corr_data=[]
for col in f1.columns:
    corr=train['loss'].corr(train[col])
    if not np.isnan(corr):
        corr_data.append([col,abs(corr)])
corr_data=pd.DataFrame(corr_data,columns=['col_name','corr']).sort_values(by='corr',ascending=False)
print(corr_data['col_name'].shape)
print(corr_data[:10])
f2=train[corr_data[:10]['col_name']]
t2=train.loss
for col in f2.columns:
    f2[col]=f2[col].fillna(f2[col].mean())
x_train1,x_test1,y_train1,y_test1=train_test_split(f2,t2,test_size=0.2,random_state=30)
print(x_train1.shape,y_train1.shape,x_test1.shape,y_test1.shape)
#lets do feature selection using permutation importance and test our dataset
from imblearn.combine import SMOTETomek
from sklearn.inspection import permutation_importance
LR=LogisticRegression().fit(x_train,y_train)
permute=permutation_importance(LR,x_test.head(2000),y_test.head(2000),random_state=20)
feature_imp=permute.importances_mean
feature_imp_df=pd.DataFrame({'features':x_test.columns,'importance':feature_imp})
print(feature_imp_df.sort_values(by='importance',ascending=False))
threshold=0.0013
selected_features=x_train.columns[feature_imp>threshold]
x_train_req=x_train[selected_features]
x_test_req=x_test[selected_features]
print(x_train_req.shape)
print(y_train.value_counts())
sm=SMOTETomek(random_state=0)
x_train_bal,y_train_bal=sm.fit_resample(x_train_req,y_train)
print(y_train_bal.value_counts())
rfc=RandomForestClassifier(n_jobs=-1).fit(x_train_bal,y_train_bal)
rfc_pred=rfc.predict(x_test_req)
print(classification_report(y_test,rfc_pred))
cmt_rfc=confusion_matrix(y_test,rfc_pred)
conf_mat=ConfusionMatrixDisplay(confusion_matrix=cmt_rfc,display_labels=[0,1])
conf_mat.plot(cmap='Blues',include_values=True)
plt.title('conf_mat of RFC after permute')
print(plt.show())
dtc=DecisionTreeClassifier().fit(x_train_bal,y_train_bal)
dtc_pred=dtc.predict(x_test_req)
print(classification_report(y_test,dtc_pred))
cmt_dtc=confusion_matrix(y_test,dtc_pred)
conf_mat=ConfusionMatrixDisplay(confusion_matrix=cmt_dtc,display_labels=[0,1])
conf_mat.plot(cmap='Blues',include_values=True)
plt.title('conf_mat of DTC after permute')
print(plt.show())
lr=LogisticRegression(n_jobs=-1).fit(x_train_bal,y_train_bal)
lr_pred=lr.predict(x_test_req)
print(classification_report(y_test,lr_pred))
cmt_lr=confusion_matrix(y_test,lr_pred)
conf_mat=ConfusionMatrixDisplay(confusion_matrix=cmt_lr,display_labels=[0,1])
conf_mat.plot(cmap='Blues',include_values=True)
plt.title('conf_mat of LR after permute')
print(plt.show())
rf_params={'n_estimators':[100,200,300],'max_features':[5,7,'auto',8,10],'min_samples_split':[2,8,15,20],'max_depth':[5,8,15,None,10],'criterion':['gini','entropy']}
cross_val=StratifiedKFold(n_splits=5)
rfc=RandomForestClassifier()
random_search=RandomizedSearchCV(rfc,param_distributions=rf_params,n_iter=20,cv=cross_val,n_jobs=-1)
random_search.fit(x_train_bal,y_train_bal)
best_params=random_search.best_estimator_
model_rf=RandomForestClassifier(n_jobs=-1,n_estimators=200,min_samples_split=8,max_features=10,criterion='entropy').fit(x_train_bal,y_train_bal)
model_rf_pred=model_rf.predict(x_test_req)
print(classification_report(y_test,model_rf_pred))
cmt_model_rf=confusion_matrix(y_test,model_rf_pred)
conf_mat=ConfusionMatrixDisplay(confusion_matrix=cmt_model_rf,display_labels=[0,1])
conf_mat.plot(cmap='Blues',include_values=True)
plt.title('conf_mat of RFC after permute and tuning')
print(plt.show())