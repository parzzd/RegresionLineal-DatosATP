import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np


data=pd.read_csv("tennis_stats.csv")
print(data.head())
print(data.info())
print(data.columns)


Variables = ['FirstServe','FirstServePointsWon',
 'FirstServeReturnPointsWon','SecondServePointsWon',
 'SecondServeReturnPointsWon','Aces','BreakPointsConverted',
 'BreakPointsFaced','BreakPointsOpportunities','BreakPointsSaved',
 'DoubleFaults','ReturnGamesPlayed','ReturnGamesWon','ReturnPointsWon',
 'ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
 'TotalServicePointsWon']


def crear_graficos(columnas):
    for columna in columnas:
        plt.title(f"grafico de victorias vs {columna}")
        plt.scatter(data[columna],data["Wins"])
        plt.xlabel(columna)
        plt.ylabel("Wins")

        plt.show()
        plt.clf()

crear_graficos(Variables)



ODS=LinearRegression()

for var in Variables:
    x=np.array(data[var]).reshape(-1,1)
    x_train,x_test,y_train,y_test=train_test_split(x,data["Wins"],train_size=0.8,test_size=0.2)

    ODS.fit(x_train,y_train)
    prediccion=ODS.predict(x_test)
    print(f"prediccion de {var}",prediccion)
    
    plt.scatter(y_test,prediccion)
    plt.show()





features = data[['BreakPointsOpportunities','FirstServeReturnPointsWon']]
winnings = data[['Winnings']]

# train, test, split the data
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)

# create and train model on training data
model = LinearRegression()
model.fit(features_train,winnings_train)

# score model on test data
print('Predicting Winnings with 2 Features Test Score:', model.score(features_test,winnings_test))

# make predictions with model
winnings_prediction = model.predict(features_test)

# plot predictions against actual winnings
plt.scatter(winnings_test,winnings_prediction, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - 2 Features')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()

## multiple features linear regression

features = data[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon','SecondServePointsWon','SecondServeReturnPointsWon','Aces','BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities','BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon','ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon','TotalServicePointsWon']]
winnings = data[['Winnings']]

features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)

model = LinearRegression()
model.fit(features_train,winnings_train)

print('Predicting Winnings with Multiple Features Test Score:', model.score(features_test,winnings_test))

winnings_prediction = model.predict(features_test)

plt.scatter(winnings_test,winnings_prediction, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - Multiple Features')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()


















