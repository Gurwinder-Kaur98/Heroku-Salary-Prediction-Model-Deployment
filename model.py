import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
df=pd.read_csv('salary_predict_dataset.csv')
df['experience'].fillna(0,inplace=True)
df['test_score'].fillna(df['test_score'].mean(),inplace=True)
df['interview_score'].fillna(df['interview_score'].mean(),inplace=True)
print(df['experience'].unique())
X=df.iloc[:,:3]
def convert_word_to_int(word):
    word_dict={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,
               'twelve':12,'zero':0,0:0,'fifteen':15,'thirteen':13}
    return word_dict[word]
X['experience']=X['experience'].apply(lambda x:convert_word_to_int(x))
y=df.iloc[:,-1]
reg=LinearRegression()
reg.fit(X,y)
reg.score(X, y)
pickle.dump(reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))


            