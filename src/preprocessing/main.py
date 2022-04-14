## loading the dataset
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

df = pd.DataFrame()
label_encoder = True
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

def loadData(dataset_path):
    global df
    df = pd.read_csv(dataset_path,
    names=['sepal_length','sepal_width','petal_length','petal_width','target'])

def handlingMissingData():
    global df
    ## 1 
    print("######### 1 #########")
    print(f"sepal_length: {df['sepal_length'].isna().sum()} - "  
        f"sepal_width: {df['sepal_width'].isna().sum()} - "
        f"petal_length: {df['petal_length'].isna().sum()} - "
        f"petal_width: {df['petal_width'].isna().sum()} - "
        f"target: {df['target'].isna().sum()}")
    print("######### /1 #########")

    ## 2
    df = df.dropna()
    print("######### 2 #########")
    print(f"sepal_length: {df['sepal_length'].isna().sum()} - "  
        f"sepal_width: {df['sepal_width'].isna().sum()} - "
        f"petal_length: {df['petal_length'].isna().sum()} - "
        f"petal_width: {df['petal_width'].isna().sum()} - "
        f"target: {df['target'].isna().sum()}")
    print("######### /2 #########")

def categoricalEncoder():
    global df,label_encoder

    ## 3
    if label_encoder:
        label_encoder = preprocessing.LabelEncoder()
        df['target'] = label_encoder.fit_transform(df['target'])
    ## 4
    else:
        ## todo
        enc = OneHotEncoder(handle_unknown='ignore')
        enc_df = pd.DataFrame(enc.fit_transform(df[['target']]).toarray())
        df = df.join(enc_df)
    # print(df)

    

def normalization():
    global df,columns
    ## 5
    print("######### 5 #########")
    print("before normalization(mean,variance)")
    var = df.var()
    mean = df.mean()
    print(f"sepal_length: {mean['sepal_length']},{var['sepal_length']} - "  
        f"sepal_width: {mean['sepal_width']},{var['sepal_width']} - "
        f"petal_length: {mean['petal_length']},{var['petal_length']} - "
        f"petal_width: {mean['petal_width']},{var['petal_width']}")

    x = df.loc[:, columns].values
    y = df.loc[:,['target']].values
    x = StandardScaler().fit_transform(x)
    y = y.transpose()[0]
    df = pd.DataFrame(x, columns =columns,dtype = float)
    s1 = pd.Series(y, name='target')
    df = pd.concat([df, s1], axis=1)
    
    print("after normalization(mean,variance)")
    var = df.var()
    mean = df.mean()
    print(f"sepal_length: {mean['sepal_length']},{var['sepal_length']} - "  
        f"sepal_width: {mean['sepal_width']},{var['sepal_width']} - "
        f"petal_length: {mean['petal_length']},{var['petal_length']} - "
        f"petal_width: {mean['petal_width']},{var['petal_width']}")
    print("######### /5 #########")
    return x,y

def pca(x,y):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                , columns = ['principal component 1', 'principal component 2'])
    s1 = pd.Series(y, name='target')
    principalDf = pd.concat([principalDf, s1], axis=1)
    # print(principalDf)
    return principalDf

def visualize(principalDf):
    pass

if __name__ == '__main__':
    dataset_path = 'assets/iris.data'
    loadData(dataset_path)
    handlingMissingData()
    categoricalEncoder()
    x,y = normalization()
    principalDf = pca(x,y)
    visualize(principalDf)

