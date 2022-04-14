## loading the dataset
import pandas as pd
from sklearn import preprocessing

df = pd.DataFrame()

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

def labelEncoding():
    global df
    ## 3
    label_encoder = preprocessing.LabelEncoder()
    df['target']= label_encoder.fit_transform(df['target'])

    

def normalization():
    pass

def pca():
    pass

def visualize():
    pass

if __name__ == '__main__':
    dataset_path = 'assets/iris.data'
    loadData(dataset_path)
    handlingMissingData()
    labelEncoding()
    normalization()
    pca()
    visualize()

