import numpy as np
import pandas as pd

#Importing Logistic Regression
from sklearn.linear_model import LogisticRegression

#Importing Accuracy Score Function
from sklearn.metrics import accuracy_score

#Importing train test split function
from sklearn.model_selection import train_test_split

#for converting strings into numeric form
from sklearn.feature_extraction.text import TfidfVectorizer

class onemodel:
    is_legit_number = False

    def __init__(self,number) -> bool:
        self.df = pd.read_csv('fraud_customers.csv')
        
        #Label Encoding
        self.df.loc[self.df['is_fraud'] == 'fraud' , 'is_fraud'] = 0
        self.df.loc[self.df['is_fraud'] == 'safe','is_fraud'] = 1

        self.X = self.df['phone_number']
        self.Y = self.df['is_fraud']
        self.X_numpy = np.asarray(self.X)
        self.X_reshape = self.X_numpy.reshape(-1,1)
        
        #Splitting the Training and Testing Data
        X_train , X_test , Y_train , Y_test = train_test_split(self.X_reshape , self.Y , test_size = 0.2, random_state = 2)

        Y_train = Y_train.astype('int')
        Y_test = Y_test.astype('int')

        model = LogisticRegression()
        model.fit(X_train , Y_train)

        #TRAINING DATA
        X_train_prediction = model.predict(X_train)
        training_data_accuracy = accuracy_score(Y_train,X_train_prediction)
        print("Accuracy of the TRAINING DATA : ",training_data_accuracy * 100,"%")

        #TESTING DATA
        X_test_prediction = model.predict(X_test)
        testing_data_accuracy = accuracy_score(Y_test , X_test_prediction)
        print("Accuracy of the TESTING DATA : ",testing_data_accuracy * 100,"%")

        input_number = [number]
        input_number_as_array = np.asarray(input_number)
        input_number_reshaped = input_number_as_array.reshape(1,-1)

        prediction = model.predict(input_number_reshaped)
        if(prediction == 1):
            print("Legitimate Mobile Number")
            self.is_legit = True
        else:
            print("WARNING ! This is not a Legitimate Number")
            self.is_legit = False

        pass