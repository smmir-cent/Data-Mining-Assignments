## loading the dataset
import pandas as pd


def loadData(dataset_path):
    df = pd.read_csv(dataset_path,
    names=['sepal_length','sepal_width','petal_length','petal_width','target'])
    return df

if __name__ == '__main__':
    dataset_path = 'assets/iris.data'
    dataset = loadData(dataset_path)
    print(dataset)
