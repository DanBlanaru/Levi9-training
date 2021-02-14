from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

import pandas as pd
# import

def imputing(train_data, method='mean', age_value = None):
    train = train_data.copy()
    train.Embarked.fillna("S",inplace = True)
    train.Fare.fillna(train.Fare.mean(),inplace = True)
    
    train.drop("Cabin", axis = 1, inplace = True)
    
    if method == 'mean':
        train.Age.fillna(train.Age.mean(), inplace = True)
    if method == 'median':
        train.Age.fillna(train.Age.median(), inplace = True)
    if age_value is not None:
        train.Age.fillna(age_value, inplace = True)
    
    train['Alone'] = 0 + (train.SibSp + train.Parch > 0)
    return train
    
    
def split_train(dataset,ratio = 0.2):
    train_data = dataset.copy()
    train_y = train_data.Survived
    train_data.drop("Survived",axis = 1,inplace = True)
    if ratio != 0:
        return model_selection.train_test_split(train_data,train_y,test_size=ratio)
    else:
        return train_data,[],train_y,[]
def transform(dataset, encoder):
    transformed_dataset = dataset.copy()
    transformed_dataset.drop("Name",axis =1, inplace = True)
    transformed_dataset.drop("PassengerId", axis =1, inplace = True)
    transformed_dataset.drop("Ticket",axis =1, inplace = True)
    if type(encoder) is OrdinalEncoder:
        transformed_dataset.Sex = encoder.fit_transform(transformed_dataset.Sex.values.reshape(-1,1))
        transformed_dataset.Embarked = encoder.fit_transform(transformed_dataset.Embarked.values.reshape(-1,1))
    if type(encoder) is OneHotEncoder:
        sex_encoded = encoder.fit_transform(transformed_dataset.Sex.values.reshape(-1,1)).toarray()
        embark_encoded = encoder.fit_transform(transformed_dataset.Embarked.values.reshape(-1,1)).toarray()
        transformed_dataset["Sex1"] = sex_encoded[:,0]
        transformed_dataset["Sex2"] = sex_encoded[:,1]
        transformed_dataset["Embarked1"] = embark_encoded[:,0]
        transformed_dataset["Embarked2"] = embark_encoded[:,1]
        transformed_dataset["Embarked3"] = embark_encoded[:,2]
        transformed_dataset.drop("Sex",axis = 1, inplace = True)
        transformed_dataset.drop("Embarked", axis = 1, inplace =  True)
#     categorical_features = ["Sex","Embarked"]
#     transformer = ColumnTransformer([("encoder", 
#                                      encoder, 
#                                      categorical_features)],
#                                      remainder="passthrough")
#     transformed_dataset = transformer.fit_transform(transformed_dataset)
#     new_names = ['Survived', 'Pclass', 'Sex', 'SibSp','Age','Parch', 'Embarked', 'Fare', 'Alone']
#     onehot_names = ['Survived', 'Pclass', 'Sex1','Sex2', 'Age', 'SibSp','Parch', 'Embarked1','Embarked2','Embarked3','Fare', 'Alone']
    return transformed_dataset

def standardize(dataset, columns = ["Age","Fare"]):
    standardized_dataset = dataset.copy()
    for column_name in columns:
        column = standardized_dataset[column_name]
        standardized_dataset[column_name] = (column - column.mean()) / column.std()
    return standardized_dataset