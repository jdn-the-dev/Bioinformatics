#imports
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import joblib


class Classifier:
    def __init__(self, file_name):
        self.file_name = file_name   

    def run(self):
        names = ['DNA_TO_CLASSIFY', 'attribute', 'sequence']
        data = pd.read_csv(self.file_name, names = names)

        print('Build our dataset using custom pandas dataframe')
        clases = data.loc[:,'DNA_TO_CLASSIFY']

        sequence = list(data.loc[:, 'sequence'])

        dic = {}
        for i, seq in enumerate(sequence):
            nucleotides = list(seq)
            nucleotides = [char for char in nucleotides if char != '\t']
            nucleotides.append(clases[i])
            
            dic[i] = nucleotides

        print('Convert Dict object into dataframe')
        df = pd.DataFrame(dic)

        print('transpose dataframe into correct format')
        df = df.transpose()
        df.rename(columns = {XXX__Length of sample__XX:'XXXXX__Sample DNA Name___XXXXX'}, inplace = True)
        

        print('Encoding')
        numerical_df = pd.get_dummies(df)
        numerical_df.drop('helitron_not-helitron', axis = 1, inplace = True)
        print(numerical_df)
        numerical_df.rename(columns = {'XXX__Fill_XXXX':'XXX__Fill__XXX'}, inplace = True)

        #Importing different classifier from sklearn
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.naive_bayes import BernoulliNB
        from sklearn.linear_model import Perceptron
        from sklearn.linear_model import SGDClassifier
        from sklearn.linear_model import PassiveAggressiveClassifier
        from sklearn.metrics import classification_report, accuracy_score

        from sklearn.model_selection import train_test_split
        X = numerical_df.drop(['heli'], axis = 1).values
        y = numerical_df['heli'].values
        

        #define a seed for reproducibility
        seed = 1

        print('Splitting data into training and testing data')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = seed)
        print(X_test[0])  
        # Define scoring method
        scoring = 'accuracy'

        print('Model building to train')
        names = ['MultinomialNB', 'BernoulliNB', 'Perceptron', 'SGDClassifier', 'PassiveAggressiveClassifier']
        Classifiers = [
            MultinomialNB(),
            BernoulliNB(),
            Perceptron(),
            SGDClassifier(),
            PassiveAggressiveClassifier(),
            ]
        models = zip(names, Classifiers)
        from sklearn.model_selection import KFold, cross_val_score

        names = []
        result = []
        for name, model in models:
            kfold = KFold(n_splits = 5, random_state = 1)
            cv_results = cross_val_score(model, X_train, y_train, cv = kfold, scoring = 'accuracy', verbose=2, n_jobs=-1)
            result.append(cv_results)
            names.append(name)
            msg = "{0}: {1} ({2})".format(name, cv_results.mean(), cv_results.std())
            print(msg)


        models = zip(names, Classifiers)
        for name, model in models:
            print("Training with: "+name)
            model.partial_fit(X_train, y_train, classes=np.unique(y_train))
            y_pred = model.predict(X_test)
            print('Exporting')
            joblib.dump(model, "Models/"+name + ".pkl")
            print(name)
            print(accuracy_score(y_test, y_pred))
            print(classification_report(y_test, y_pred))

directory = 'data/'
for entry in os.scandir(directory):
    dnaClassify = Classifier(entry.path) #Input Files goes here
    dnaClassify.run()
