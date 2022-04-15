## loading the dataset
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


df = pd.DataFrame()
box_plot_df = pd.DataFrame()


columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

def loadData(dataset_path):
    global df
    df = pd.read_csv(dataset_path,
    names=['sepal_length','sepal_width','petal_length','petal_width','target'])

def handlingMissingData():
    global df,box_plot_df
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
    df = df.reset_index(drop=True)
    box_plot_df = df.copy()
    print("######### 2 #########")
    print(f"sepal_length: {df['sepal_length'].isna().sum()} - "  
        f"sepal_width: {df['sepal_width'].isna().sum()} - "
        f"petal_length: {df['petal_length'].isna().sum()} - "
        f"petal_width: {df['petal_width'].isna().sum()} - "
        f"target: {df['target'].isna().sum()}")
    print("######### /2 #########")

def categoricalEncoder():
    global df
    ## 3
    label_encoder = preprocessing.LabelEncoder()
    df['target'] = label_encoder.fit_transform(df['target'])
    print("######### 3 #########")
    print(label_encoder.inverse_transform([0]))
    print(label_encoder.inverse_transform([1]))
    print(label_encoder.inverse_transform([2]))
    print("######### /3 #########")

    ## 4
    print("######### 4 #########")
    enc = preprocessing.OneHotEncoder()
    df_enc = enc.fit_transform(df[['target']].values.reshape(-1,1)).toarray()
    types = ('Iris-setosa','Iris-versicolor','Iris-virginica')
    df_enc = pd.DataFrame(df_enc,columns=types)
    print(df_enc)
    print("######### /4 #########")
   
    

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
    print("######### 6 #########")
    print(principalDf)
    print("######### /6 #########")
    return principalDf

def visualize(principalDf):
    global box_plot_df
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'g', 'b']
    for _, tuple in principalDf.groupby("target"):
        plt.scatter(tuple["principal component 1"], tuple["principal component 2"])
    ax.legend(targets)
    ax.grid()
    ax.plot()
    fig.savefig("main.png")
    plt.cla()
    plt.figure(figsize = (10,10))
    boxplot = box_plot_df.boxplot(column=['sepal_length','sepal_width','petal_length','petal_width'])
    # fig.savefig("boxplot.png")
    plt.savefig("boxplot.png")
    

if __name__ == '__main__':
    dataset_path = 'assets/iris.data'
    loadData(dataset_path)
    handlingMissingData()
    categoricalEncoder()
    x,y = normalization()
    principalDf = pca(x,y)
    visualize(principalDf)

