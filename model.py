# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


import pandas as pd

# Sample Data
data = {
    "experience": ["zero", "zero", "five", "two", "seven", "three", "ten", "eleven"],
    "test_score": [8, 6, 6, 10, 9, 7, 7.85714, 7],
    "interview_score": [9, 6, 7, 10, 6, 10, 7, 8],
    "salary": [50000, 45000, 60000, 65000, 70000, 62000, 72000, 80000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV file correctly
df.to_csv("hiring_data.csv", index=False)


dataset = pd.read_csv('hiring_data.csv')


X = dataset.iloc[:, :3]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()



#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))