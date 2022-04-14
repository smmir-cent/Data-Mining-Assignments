## loading the dataset
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

df = pd.DataFrame()
label_encoder = True

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
    global df
    ## 5
    print("######### 5 #########")
    print("before normalization(mean,variance)")
    var = df.var()
    mean = df.mean()
    print(f"sepal_length: {mean['sepal_length']},{var['sepal_length']} - "  
        f"sepal_width: {mean['sepal_width']},{var['sepal_width']} - "
        f"petal_length: {mean['petal_length']},{var['petal_length']} - "
        f"petal_width: {mean['petal_width']},{var['petal_width']}")
    col_names = ['sepal_length','sepal_width','petal_length','petal_width']
    features = df[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    df[col_names] = features
    # print(df)
    print("after normalization(mean,variance)")
    var = df.var()
    mean = df.mean()
    print(f"sepal_length: {mean['sepal_length']},{var['sepal_length']} - "  
        f"sepal_width: {mean['sepal_width']},{var['sepal_width']} - "
        f"petal_length: {mean['petal_length']},{var['petal_length']} - "
        f"petal_width: {mean['petal_width']},{var['petal_width']}")
    print("######### /5 #########")

def pca():
    pass

def visualize():
    pass

if __name__ == '__main__':
    dataset_path = 'assets/iris.data'
    loadData(dataset_path)
    handlingMissingData()
    categoricalEncoder()
    normalization()
    pca()
    visualize()

