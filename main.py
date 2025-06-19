from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, RidgeClassifier, LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,SVR
import joblib
from feature_engine.outliers import Winsorizer

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler  # Optional, for scaling input features
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os, joblib


global filename
global regressor
global X, y, X_train, X_test, y_train, y_test ,Predictions
global df, df2, sc
global le, labels


# Global Variables
filename = ""
dataset = None
X, y, X_train, X_test, y_train, y_test = None, None, None, None, None, None
mae_list, mse_list, rmse_list, r2_list = [], [], [], []

# Upload Dataset
def upload():
    global filename, dataset
    filename = filedialog.askopenfilename()
    text.delete('1.0', END)
    text.insert(END, f"{filename} Loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END, str(dataset.head()) + "\n")

# Data Preprocessing
def preprocess():
    global dataset, X, y, X_train, X_test, y_train, y_test
    dataset['DateTime'] = pd.to_datetime(dataset['DateTime'], format='%d-%m-%Y %H:%M')
    dataset['Month'] = dataset['DateTime'].dt.month
    dataset['Hour'] = dataset['DateTime'].dt.hour
    dataset['Day'] = dataset['DateTime'].dt.day
    dataset['Year'] = dataset['DateTime'].dt.year
    dataset.drop(columns=['DateTime'], inplace=True)
    
    X = dataset.drop('TotalUsage', axis=1)
    y = dataset['TotalUsage']
    for column in dataset.columns:
        plt.figure(figsize=(6, 4))
        dataset.boxplot(column=[column])
        plt.title('Box Plot of {}'.format(column))
        plt.ylabel('Value')
        plt.xlabel('{}'.format(column))
        plt.grid(True)
        plt.show()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)
    text.insert(END, f"Training Records: {len(X_train)} | Testing Records: {len(X_test)}\n\n")

# Performance Metrics
def PerformanceMetrics(algorithm, predict, testY):
    testY = testY.astype('int')
    predict = predict.astype('int')
    
    mse = mean_squared_error(testY, predict)
    mae = mean_absolute_error(testY, predict)
    r2 = r2_score(testY, predict) * 100
    
    # Append calculated metrics to lists
    mae_list.append(mae)
    mse_list.append(mse)
    rmse = np.sqrt(mse)
    r2_list.append(r2)
    text.insert(END, f"{algorithm} Metrics:\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR2 Score: {r2:.2f}\n\n")
    # Create a scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(predict, testY, color='blue')
    plt.plot([min(testY), max(testY)], [min(testY), max(testY)], linestyle='--', color='red', lw=2)  # Identity line
    # Set labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Regression Performance')
    plt.show()
# Support Vector Regressor (SVR)
def SVR_model():
    path = 'SVR_model.pkl'
    if os.path.exists(path):
        model = joblib.load(path)
    else:
        model = SVR()
        model.fit(X_train, y_train)
        joblib.dump(model, path)
    y_pred = model.predict(X_test)
    PerformanceMetrics("SVR", y_test, y_pred)

# Random Forest Regressor
global model
def RandomForest_model():
    global model
    path = 'RF_model.pkl'
    if os.path.exists(path):
        model = joblib.load(path)
    else:
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        joblib.dump(model, path)
    y_pred = model.predict(X_test)
    PerformanceMetrics("RandomForest", y_test, y_pred)
# Performance Graph

    
   

def predict():
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    test=pd.read_csv(filename)
    test['DateTime'] = pd.to_datetime(test['DateTime'], format='%d-%m-%Y %H:%M')

    test['Month'] = test['DateTime'].dt.month
    test['Hour'] = test['DateTime'].dt.hour
    test['Minute'] = test['DateTime'].dt.minute
    test['Day'] = test['DateTime'].dt.day
    test['Year'] = test['DateTime'].dt.year

    test.drop(columns=['DateTime'], inplace=True)

    # Make predictions on the selected test data
    predictions = model.predict(test)

    # Loop through each prediction and print the corresponding row and prediction value
    for i, prediction in enumerate(predictions):
        text.insert(END, test.iloc[i])
        text.insert(END, "Row {}: Prediction =======> {}".format(i, prediction))



    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')

def close():
  main.destroy()

from tkinter import *
main = Tk()
main.title("Energy consumption forecasting")
main.geometry("1000x850")
main.config(bg='light steel blue')

font = ('Times New Roman', 15, 'bold')
title = Label(main, text='Energy Consumption Forecasting in Smart Grids', 
              justify=CENTER, bg='deepskyblue', fg='white')  # Updated colors for title
title.config(font=font, height=2, width=100)
title.pack(pady=10)

button_frame = Frame(main, bg='light steel blue')
button_frame.pack(pady=20)

font1 = ('Times New Roman', 12, 'bold')  # Changed button font to Times New Roman

# Helper function to create buttons in a grid layout
def create_button(text, command, row, col):
    Button(button_frame, text=text, command=command, bg='lightblue', fg='black', 
           activebackground='lavender', font=font1, width=25).grid(row=row, column=col, padx=15, pady=15)

# First row of buttons
create_button("Upload Productivity Dataset", upload, 0, 0)
create_button("Data Analysis and Preprocessing", preprocess, 0, 1)

# Second row of buttons
create_button("SVR", SVR_model, 1, 0)
create_button("rfr Model",  RandomForest_model, 1, 1)


# Third row of buttons
create_button("Prediction on Test Data", predict, 2, 0)
create_button("Close Application", close, 2, 1)
# Optionally add more buttons here if needed

# Text Box with Scrollbar (placed at the bottom for displaying results/logs)
text_frame = Frame(main, bg='lavender')
text_frame.pack(pady=20)  # Padding for spacing

# Updated Text Box and Scrollbar appearance
text = Text(text_frame, height=25, width=125, wrap=WORD, bg='white', fg='black', font=('Times New Roman', 12))  # Changed font to Times New Roman
scroll = Scrollbar(text_frame, command=text.yview)
text.configure(yscrollcommand=scroll.set)

text.pack(side=LEFT, fill=BOTH, expand=True)
scroll.pack(side=RIGHT, fill=Y)

# Run the main loop
main.mainloop()
