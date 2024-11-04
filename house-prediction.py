import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import PowerTransformer
from scipy.stats import levene
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression,f_classif,chi2,mutual_info_classif,mutual_info_regression
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import random 



df=pd.read_csv('hous_pricing\\train.csv')
print(df.info())
df=df.set_index('Id')   
report=ProfileReport(df,title='factors',type_schema = {"MSSubClass": "categorical", "OverallQual": "categorical","OverallCond": "categorical",
                                                                        "BsmtFullBath":"numeric", "BsmtHalfBath":"numeric", "FullBath":"numeric",
                                                                    "HalfBath":"numeric", "Kitchen":"numeric", "Fireplaces":"numeric",
                                                                        "GarageCars":"numeric","YrSold":"numeric"})
#report.to_file('hous_pricing\price-eda.html')

column_list=['Alley','BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu',
             'GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']


df[column_list]=df[column_list].replace(np.nan,'non-existent')
df.info()


def frequncyTble(data):
    data=data.astype(str)
    categories,counts=np.unique(data,return_counts=True)
    result=pd.DataFrame({'categories':categories,'counts':counts})
    return result

# for i in df.columns:
#     print(frequncyTble(df[i]))


target=df.SalePrice
inputs=df.drop('SalePrice',axis=1)

X_train,X_test,Y_train,Y_test=train_test_split(inputs,target,test_size=0.3,random_state=42)




def pre_proc(data):

    processed_data=data.copy()
    bins=[-np.inf,20,60,100,140,180,np.inf]
    processed_data['MSSubClass']=pd.cut(processed_data['MSSubClass'],bins=bins,labels=[1,2,3,4,5,6])
    print(processed_data['MSSubClass'])

    
    # max_minInspector={'':()}
    #nasazegari, khareje baze



    processed_data['LotShape'] = processed_data['LotShape'].replace(['IR1','IR2','IR3'],'IR')
    processed_data['LandContour'] = processed_data['LandContour'].replace(['Bnk','HLS','Low'],'Not Lvl')
    processed_data['LotConfig'] = processed_data['LotConfig'].replace(['CulDSac','FR2','FR3'],'Others')
    processed_data['LandSlope'] = processed_data['LandSlope'].replace(['Mod','Sev'],'Not Gtl')
    processed_data['Condition1'] = processed_data['Condition1'].apply(lambda x: 'Norm' if x == 'Norm' else 'Not Norm')
    processed_data['Condition2'] = processed_data['Condition2'].apply(lambda x: 'Norm' if x == 'Norm' else 'Not Norm')
    processed_data['RoofStyle'] = processed_data['RoofStyle'].apply(lambda x: 'Gable' if x == 'Gable' else 'Not Gable')
    processed_data['MasVnrType'] = processed_data['MasVnrType'].apply(lambda x: 'Stone' if x == 'Stone' else 'Brick')
    processed_data['ExterQual'] = processed_data['ExterQual'].replace({'non-existent': '0', 'Gd': '3', 'Ex': '3', 'Fa': '1', 'Po': '1', 'TA': '2'})
    processed_data['ExterCond'] = processed_data['ExterCond'].replace({'non-existent': '0', 'Gd': '3', 'Ex': '3', 'Fa': '1', 'Po': '1', 'TA': '2'})
    processed_data['BsmtQual'] = processed_data['BsmtQual'].replace({'non-existent': '0', 'Gd': '3', 'Ex': '3', 'Fa': '1', 'Po': '1', 'TA': '2'})
    processed_data['BsmtCond'] = processed_data['BsmtCond'].replace({'non-existent': '0', 'Gd': '3', 'Ex': '3', 'Fa': '1', 'Po': '1', 'TA': '2'})
    processed_data['Heating'] = processed_data['Heating'].apply(lambda x: 'GasA' if x == 'GasA' else 'Others')
    processed_data['HeatingQC'] = processed_data['HeatingQC'].replace({'non-existent': '0', 'Gd': '3', 'Ex': '3', 'Fa': '1', 'Po': '1', 'TA': '2'})
    processed_data['Electrical'] = processed_data['Electrical'].apply(lambda x: 'Standard' if x == 'SBrkr' else 'Not Standard')
    processed_data['KitchenQual'] = processed_data['KitchenQual'].replace({'non-existent': '0', 'Gd': '3', 'Ex': '3', 'Fa': '1', 'Po': '1', 'TA': '2'})
    processed_data['Functional'] = processed_data['Functional'].apply(lambda x: 'Typ' if x == 'Typ' else 'Not Typ')
    processed_data['FireplaceQu'] = processed_data['FireplaceQu'].replace({'non-existent': '0', 'Gd': '3', 'Ex': '3', 'Fa': '1', 'Po': '1', 'TA': '2'})
    processed_data['GarageQual'] = processed_data['GarageQual'].replace({'non-existent': '0', 'Gd': '3', 'Ex': '3', 'Fa': '1', 'Po': '1', 'TA': '2'})
    processed_data['GarageCond'] = processed_data['GarageCond'].replace({'non-existent': '0', 'Gd': '3', 'Ex': '3', 'Fa': '1', 'Po': '1', 'TA': '2'})
    processed_data['BsmtExposure'] = processed_data['BsmtExposure'].replace({'non-existent': '0', 'No': '1', 'Mn': '2', 'Av': '3', 'Gd': '4'})
    processed_data['BsmtFinType1'] = processed_data['BsmtFinType1'].replace({'non-existent': '0', 'Unf': '1', 'LwQ': '2', 'Rec': '3', 'BLQ': '4', 'ALQ': '5', 'GLQ': '6'})
    processed_data['BsmtFinType2'] = processed_data['BsmtFinType2'].replace({'non-existent': '0', 'Unf': '1', 'LwQ': '2', 'Rec': '3', 'BLQ': '4', 'ALQ': '5', 'GLQ': '6'})
    processed_data['SaleType'] = processed_data['SaleType'].apply(lambda x: 'WD' if x == 'WD' else 'Not WD')
    processed_data['SaleCondition'] = processed_data['SaleCondition'].apply(lambda x: 'Normal' if x == 'Normal' else 'Not Normal')

    return processed_data



def feature_screening(data, min_cv=0.1, mode_threshold=95, distinct_threshold=90):

    processed_data=data.copy()

    cat_col=processed_data.select_dtypes(include=['category','object']).columns

    cont_col=processed_data.select_dtypes(exclude=['object','category']).columns
    #cv: std/mean >0.1


    cv=processed_data[cont_col].std()/processed_data[cont_col].mean()
    cvIndx=cv[cv<0.1].index.tolist()

    #low variance

    low_variance=processed_data[cat_col].apply(lambda x:x.value_counts().max()/len(x))
    lVarIndx=low_variance[low_variance>0.94].index.tolist()


    # high variance

    high_variance=processed_data[cat_col].apply(lambda x:x.nunique()/len(x))
    hVarindx=high_variance[high_variance>0.95].index.tolist()
    dropedcol=set(cvIndx+lVarIndx+hVarindx)
    print(dropedcol)
    
    return list(dropedcol)



# outlier -isolationforest

def outlier_handler(data,contamination):
    data2=data.copy()

    cat_col = data2.select_dtypes(include=['object','category']).columns.tolist()
    cont_col = data2.select_dtypes(exclude=['object','category']).columns.tolist()

    for col in data2.columns:
        if col in cont_col:
            data2[col]=data2[col].fillna(data2[col].mean())
        elif col in cat_col :
            data2[col]=data2[col].fillna(data2[col].mode().iloc[0])

    

    ordinal = ['MSSubClass', 'ExterQual', 'ExterCond','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
            'BsmtFinType2','HeatingQC','KitchenQual','FireplaceQu','GarageQual', 'GarageCond']

    nominal = ['MSZoning','LotShape','LandContour','LotConfig','Neighborhood', 'Condition1','BldgType',
            'HouseStyle', 'RoofStyle','Exterior1st', 'Exterior2nd', 'MasVnrType','Foundation','CentralAir',
            'Functional','GarageType','GarageFinish','PavedDrive','Fence','SaleType', 'SaleCondition']

    oe=OrdinalEncoder()
    data2[ordinal]=oe.fit_transform(data2[ordinal])
   
    ohe=OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore')
    # khoroojie one_hot_encoded numpy ndarray hast 
    one_hot_encoded=ohe.fit_transform(data2[nominal])
    one_hot_encoded_df=pd.DataFrame(one_hot_encoded,columns=ohe.get_feature_names_out())

    #baraye concat bayad 2tashun df bashan!
    iso_input_encoded=pd.concat((one_hot_encoded_df,data2[ordinal].reset_index(),data2[cont_col].reset_index()),axis=1)
    print(iso_input_encoded)
    ss=StandardScaler()
    iso_input_encodeScaled=ss.fit_transform(iso_input_encoded)

    # for i in data2.columns:
    #     if data2[i].isna().any():
    #         df_TF=data2[i].isna()
    #         res=df_TF[df_TF==True].index
    #         print(res)
    #     else:
    #         continue

    clf=IsolationForest(contamination=contamination,random_state=32)
    clf.fit(iso_input_encodeScaled)
    outliers=clf.predict(iso_input_encodeScaled)
    iso_input_encoded['outliers']=outliers
    #code paiin eshtebahe chon aar ba index kar konim ke dartule masir reset shode sakhte ba filede Id kar mikoniam
    # kheili rahat tare 
    #outlier_index = iso_input_encoded[iso_input_encoded['outliers'] == -1].index
    outlier_index = iso_input_encoded[iso_input_encoded['outliers'] == -1]['Id']
    print('&&&&&&&&&&&&&&&&&&&&&&&&&',iso_input_encoded)
    return outlier_index





X_train=pre_proc(X_train)
X_test=pre_proc(X_test)
print('*************************************',len(X_test.columns),len(X_train.columns))
dropedList=feature_screening(X_train)
X_train=X_train.drop(dropedList,axis=1)
X_test=X_test.drop(dropedList,axis=1)

outlier=outlier_handler(X_train,0.02)
# print(len(outlier))
# res=X_train_screened.iloc[:,0].isin(outlier)
# print(len(res[res==True]))

data = pd.concat((X_train,Y_train),axis=1)
data=data.drop(outlier.iloc[:,0].to_list(),axis=0)
X_train=data.iloc[:,:-1]
Y_train=data.iloc[:,-1]

#X_train_screened=X_train_screened[~X_train_screened['Id'].isin(outlier)]


def missing_handler(data,test,missminRow=35):

    processed_data=data.copy()
    processed_data['numberOfmissng']=processed_data.isnull().sum(axis=1)
    index_of_missing=processed_data[processed_data['numberOfmissng']>missminRow].index
    processed_data=processed_data.drop(index_of_missing.to_list(),axis=0)
    
    percentageOfCol=processed_data.isnull().sum()/len(processed_data['numberOfmissng'])
    processed_data=processed_data.drop(percentageOfCol[percentageOfCol>0.5].index)
    test=test.drop(percentageOfCol[percentageOfCol>0.5].index)

    # percentage=len(missingdata[missingdata>0])/len(missingdata)
    res=processed_data['numberOfmissng'].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    res2=processed_data.isnull().sum(axis=0)/len(processed_data['numberOfmissng'])
    return data,test


X_train,X_test=missing_handler(X_train,X_test)



def missingImputer(data,test):

    listOfKnn = ['LotFrontage']
    listImpcat = ['BsmtCond']
    listImpCont = ['MasVnrArea']

    knnimp = KNNImputer(n_neighbors=5,weights='distance')
    data[listOfKnn]=knnimp.fit_transform(data[listOfKnn])
    test[listOfKnn]=knnimp.transform(test[listOfKnn])


    simpImCat = SimpleImputer(strategy='most_frequent')
    data[listImpcat]=simpImCat.fit_transform(data[listImpcat])
    test[listImpcat]=simpImCat.transform(test[listImpcat])

    simpImCont=SimpleImputer(strategy='mean')
    data[listImpCont]=simpImCont.fit_transform(data[listImpCont])
    test[listImpCont]=simpImCont.transform(test[listImpCont])

    return data,test



X_train,X_test=missingImputer(X_train,X_test)
print('missImp',X_train,X_test)
# print(missing_handler(X_train,X_test,35))


def discritizer(data,test):

    zero_inflated_list =['MasVnrArea', 'BsmtFinSF1', '2ndFlrSF', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF'] 
    zero_drop_list = ['BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
    kbin=KBinsDiscretizer(n_bins=5,encode='ordinal',strategy='kmeans')
    data=data.drop(zero_drop_list,axis=1)
    test=test.drop(zero_drop_list,axis=1)
    data[zero_inflated_list]=kbin.fit_transform(data[zero_inflated_list])
    test[zero_inflated_list]=kbin.transform(test[zero_inflated_list])
    data[zero_inflated_list]=data[zero_inflated_list].astype('category')
    test[zero_inflated_list]=test[zero_inflated_list].astype('category')
    # for var in zero_inflated_list:
    #     print(frequncyTble(data[var]))
    for col in zero_inflated_list : 
        print(col,frequncyTble(data[col]))

    return data,test
    
X_train,X_test=discritizer(X_train,X_test)
print('disc',X_train,X_test)



def trandformer(data,test):

    transform_list = ['LotFrontage','LotArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea']    
    
    for var in transform_list:
        if (data[var]<=0).any():

            transforming=PowerTransformer(method='yeo-johnson',standardize=False)
        else:
            transforming=PowerTransformer(method='box-cox',standardize=False)

        data[var]=transforming.fit_transform(data[[var]])
        test[var]=transforming.transform(test[[var]])

    return data,test

X_train,X_test = trandformer(X_train,X_test)
print('transform',X_train,X_test)


# print(len(X_train.columns),len(X_test.columns))
# rs=X_test.columns.isin(X_train.columns)
# indx=np.where(rs==False)
# print(X_test.columns[indx])

def feature_selection(data):

    catcol = data.select_dtypes(include=['object','category']).columns.to_list()
    contCol = data.drop(catcol,axis=1).columns.to_list()
    data[catcol] = data[catcol].astype('category')

    oe = OrdinalEncoder()
    data[catcol] = oe.fit_transform(data[catcol])

    selectFtur = SelectKBest(score_func=f_regression,k='all')
    selectFtur2 = SelectKBest(score_func=f_classif,k='all')
    mutCatSelectfutur=SelectKBest(score_func=mutual_info_classif,k='all')
    mutContSelectfutur=SelectKBest(score_func=mutual_info_regression,k='all')
    selectFtur2.fit(data[catcol],Y_train)
    selectFtur.fit(data[contCol],Y_train)
    mutCatSelectfutur.fit(data[catcol],Y_train)
    mutContSelectfutur.fit(data[contCol],Y_train)
    
    dfContinfogain = pd.DataFrame({'score':mutContSelectfutur.scores_,'name':mutContSelectfutur.get_feature_names_out().tolist()})
    dfCatinfogain = pd.DataFrame({'score':mutCatSelectfutur.scores_,'name':mutCatSelectfutur.get_feature_names_out().tolist()})
    # selectFtur=SelectKBest(score_func=mutual_info_regression,k='all')
    resCat = pd.DataFrame({'score':selectFtur2.scores_,'name':selectFtur2.get_feature_names_out().tolist()})
    resCont = pd.DataFrame({'score': selectFtur.scores_,'name':selectFtur.get_feature_names_out().tolist()})

    res1 = resCont.sort_values(by = 'score',ascending=False)
    res2 = resCat.sort_values(by = 'score',ascending=False)
    resMut1 = dfContinfogain.sort_values(by='score',ascending=False)
    resMut2 = dfCatinfogain.sort_values(by='score',ascending=False)


    
    
    # plt.bar([i for i in range(len(res1.score))],res1.score)
    # plt.show()
    # plt.bar([i for i in range(len(res2))],res2.score)
    # plt.show()
    # plt.bar([i for i in range (len(resMut1.score))],resMut1.score)
    # plt.show()
    # plt.bar([i for i in range(len(resMut2.score))],resMut2.score)
    # plt.show()
    # normalcatList=[]
    # abnormalcatList=[]
    # for col in catcol:
    #     groups = [Y_train[X_train[col] == label] for label in np.unique(X_train[col])]
    #     stat,pval=levene(*groups)
    #     if pval >0.05:
    #         normalcatList.append(col)
    #     else:
    #         abnormalcatList.append(col)
    # dfRes=pd.DataFrame({'normal':pd.Series(normalcatList),'abnormal':pd.Series(abnormalcatList)})

    return set(res1.iloc[:13,1].to_list()+res2.iloc[:12,1].to_list()+resMut1.iloc[:21,1].to_list()+resMut2.iloc[:21,1].to_list())

feature_selected = list(feature_selection(X_train))
print('********************sahba NOTICE!',feature_selected)
# X_train=X_train[feature_selected]
# print(X_train.shape)
# print(X_train)

def embeded_selection(data,target):

    nominal = ['MSZoning','Alley','LotShape','LandContour','LotConfig','Neighborhood', 'Condition1','BldgType',
           'HouseStyle', 'RoofStyle','Exterior1st', 'Exterior2nd', 'MasVnrType','Foundation','CentralAir','Electrical',
           'Functional','GarageType','GarageFinish','PavedDrive','Fence','SaleType', 'SaleCondition']   

    other  = [i for i in X_train.columns if i not in nominal]

   
    ohe=OneHotEncoder(sparse_output=False,handle_unknown='ignore',drop='first')
    X_ohe_encoded=ohe.fit_transform(data[nominal])
    df_nominal=pd.DataFrame(X_ohe_encoded,columns=ohe.get_feature_names_out())
    x_train_encoded=pd.concat([df_nominal,data[other].reset_index()],axis=1)
    x_train_encoded=x_train_encoded.set_index('Id')
    
    scaler=MinMaxScaler()
    x_train_scaled=scaler.fit_transform(x_train_encoded)
    X_Train=pd.DataFrame(x_train_scaled,columns=x_train_encoded.columns)
    selector=RFECV(estimator=DecisionTreeRegressor(random_state=29),cv=5,n_jobs=-1,min_features_to_select=10,step=1)
    selector.fit(X_Train,target)
    res1=selector.get_support()
    res2=selector.n_features_
    res3=selector.get_feature_names_out()
    res4=X_Train[res3]
    return res1,res2,res3,res4


print(embeded_selection(X_train,Y_train))


def feature_extract(data,target):
    print('FDFDFD',data.columns)
    categorical = ['MSSubClass', 'MSZoning', 'Alley', 'LotShape', 'LandContour', 'LotConfig' , 'Neighborhood', 
               'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
               'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
               'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 
               'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'Fence', 'SaleType', 
               'SaleCondition', 'MasVnrArea', 'BsmtFinSF1', '2ndFlrSF', 'GarageArea', 'WoodDeckSF', 
               'OpenPorchSF']
    
    
    continues = data.drop(categorical,axis=1).columns.to_list()
    
    scaler = StandardScaler()
    tempArray = scaler.fit_transform(data[continues])
    data = pd.DataFrame(tempArray , columns=scaler.get_feature_names_out())
    corr_matrix = data[continues].corr()
    sns.heatmap(corr_matrix)
    plt.show()

    pca_fe = PCA(n_components=10,random_state=29)
    pca_fe.fit(data[continues])
    res_name = pd.DataFrame([f'pca_{i}'for i in range(len(pca_fe.explained_variance_))],columns=['name'])
    res_value = pd.DataFrame(pca_fe.explained_variance_,columns=['value'])

    res_percent = pd.DataFrame(pca_fe.explained_variance_ratio_,columns=['percent'])
    res_percent_comu = pd.DataFrame(pca_fe.explained_variance_ratio_.cumsum(),columns=['comulative'])
    final_res = pd.concat([res_name,res_value,res_percent,res_percent_comu],axis=1)
    
    weights = pd.DataFrame(pca_fe.components_,columns=data.columns,index=final_res.name).transpose().sort_values(by='pca_1')
    pca_tempSeries = pca_fe.transform(X_train[continues])
    pca_extractedxTrain = pd.DataFrame(pca_tempSeries,columns=res_name.name)

    total = pd.concat((target.reset_index(drop=True),pca_extractedxTrain),axis=1)
    
    total.iloc[:,0] = total.iloc[:,0].apply(lambda x : 4 if x>=300000 else 3 if 165000<x<=300000 else 2 if 100000<x<165000 else 1)
    sns.pairplot(total,hue='SalePrice')
    plt.show()
    print(weights)
    print(total.corr())
    return total

    
# X_train=feature_extract(X_train,Y_train)
# X_train=X_train.drop(['pca_1','pca_2','pca_3','pca_4','pca_7','pca_8','pca_9'],axis=1)
# X_train=X_train.reindex(columns=['pca_0','pca_5','pca_6','SalePrice'])
# # X_train=X_train.rename(columns=[''])
# print(X_train)

def kernel_PCA(data,gam,target):
    
    kpca = KernelPCA(n_components=4,kernel='rbf',gamma=gam,remove_zero_eig=True)   
    tempKpca = kpca.fit_transform(data)
    cols = [f'pca_{i}'for i in range(tempKpca.shape[1])] 
    kpca_X = pd.DataFrame(tempKpca,columns=cols)

    kpca_Xtrian = pd.concat((kpca_X,target.reset_index(drop=True)) , axis=1)
    kpca_Xtrian.iloc[:,0] = kpca_Xtrian.iloc[:,0].apply(lambda x : 4 if x>=300000 else 3 if 165000<x<=300000 else 2 if 100000<x<165000 else 1)
    sns.pairplot(kpca_Xtrian,hue='SalePrice')
    return kpca_Xtrian



def ldaFe(data,target):
    categorical = ['MSSubClass', 'MSZoning', 'Alley', 'LotShape', 'LandContour', 'LotConfig' , 'Neighborhood', 
               'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
               'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
               'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 
               'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'Fence', 'SaleType', 
               'SaleCondition', 'MasVnrArea', 'BsmtFinSF1', '2ndFlrSF', 'GarageArea', 'WoodDeckSF', 
               'OpenPorchSF']
    continues = data.drop(categorical,axis=1).columns.tolist()
    scaler = StandardScaler()
    data[continues] = scaler.fit_transform(data[continues])
    target_cat=target.apply(lambda x:5 if x>=300000 else 4 if 220000<=x<300000 else 3 if 140000<=x<220000 else 2 if 140000<x<=60000 else 1 )
    # n_compnant=min(n_features , class Target -1 )
    lda=LinearDiscriminantAnalysis(n_components=3)
    lda.fit(data[continues],target_cat)
    eigVal = lda.explained_variance_ratio_
    plt.plot(eigVal)
    plt.show()
    
    lda2=LinearDiscriminantAnalysis(n_components=1)
    lda2.fit(data[continues],target_cat)
    xtrain_tempseries = lda2.transform(data[continues])
    X_train_fe = pd.DataFrame(xtrain_tempseries,columns=['comp1'])

    total=pd.concat((X_train_fe,target_cat.reset_index(drop=True)),axis=1)
    sns.pairplot(total,hue='SalePrice')
    plt.show()


def normalSampling(xtr,ytr):
    data=pd.concat((xtr,ytr),axis=1)

    sample_size=160
    frac_size=0.16
    data=data.sample(n=sample_size,random_state=42)
    # data=data.sample(frac=frac_size,random_state=42)
    return data.shape   

# print(normalSampling(X_train,Y_train))


def systematicSampling(xtr,ytr):
    total=pd.concat((xtr,ytr),axis=1)
    total.sort_values('SalePrice')

    sampleSize=100
    interval=int(len(total)/sampleSize)
    random.seed(110)
    startindx=random.randint(0,interval-1)
    data=total.iloc[startindx::interval,:]
    return data

# print(systematicSampling(X_train,Y_train))
# if __name__ == '__main__':
# # print(kernel_PCA(X_train,0.8,Y_train))
#     print(ldaFe(X_train,Y_train))
 
def stratifiedSampling(xtr,ytr):
    total = pd.concat((xtr,ytr),axis=1)
    stra_df=total['Neighborhood'].value_counts().sort_values(ascending=False)
    ratio = 0.2
    sample_df = pd.DataFrame()
    for state,count in stra_df.items():
        # sampling=total[total['Neighborhood']==state].sample(frac=ratio)
        # sample_df=pd.concat((sampling,sample_df),axis=0)

        sample_size = 100
        
        intervalSample = int(len(xtr)/sample_size)
        startsampling = random.randint(0,intervalSample-1)
        sampling = total[total['Neighborhood']==state].iloc[startsampling::intervalSample,:]
        sample_df = pd.concat((sample_df,sampling),axis=0)
    return sample_df.shape


print(stratifiedSampling(X_train,Y_train))


def clusterSampling(xtr,ytr):

    total=pd.concat((xtr,ytr),axis=1)
    samples=list(total['Neighborhood'].unique())
    k=5
    clusterList=random.sample(samples,k)
    result_df=pd.DataFrame()
    sampled_df=total[total['Neighborhood'].isin(clusterList)]
    result_df=pd.concat((sampled_df,result_df),axis=1)
    return result_df.shape

print(clusterSampling(X_train,Y_train))