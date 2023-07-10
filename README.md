# Energy Consumption Prediction Project

This project aims to predict energy consumption from appliances using time series analysis techniques. The project utilizes three different approaches: deep learning techniques with data windowing and deep learning models, SARIMAX statistical technique, and Facebook's Prophet model. The predictions are based on historical data and various features related to energy usage.

## Table of Contents
- [Description](#description)
- [Data](#data)
- [Tools and Skills](#tools-and-skills)
- [Results](#results)
- [Deep Learning Techniques](#deep-learning-techniques)
- [SARIMAX Statistical Technique](#sarimax-statistical-technique)
- [Facebook Prophet Model](#facebook-prophet-model)

## Description
This data science project focuses on predicting energy consumption from appliances using time series analysis techniques. The goal is to develop accurate models that can forecast energy usage based on historical data. From this, appliance usage can be tailored to predicted ranges and costs can be derived. Code was inspired by Marco Peixeiro in his Udemy course 'Applied Time Series Analysis in Python'. Note that paramweter optimization functions were run in separate jupyter notebook rather than google colab due to connectivity retraints.

## Data
The data used in this project consists of historical energy consumption records from appliances. It includes information such as timestamps, weather conditions, and various other features that can influence energy usage. The dataset is preprocessed and prepared for time series analysis. The dataset can be found at the UCI Machine Learning Repository at this link: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction

## Tools and Skills
Work on this project involved:
- Python 
- Libraries: TensorFlow, Keras, statsmodels, Prophet
- Jupyter Notebook, google colab, or any Python IDE

The following skills were utilized:
- Time series analysis
- Deep learning techniques
- Statistical modeling
- Python programming

## Results
The results show a few conclusions:
1. The overall energy consumption of the appliances is decreasing over time.
2. There is a daily and weekly seasonality to the consumption, with peak times being evening and peak days being Saturdays
3. The models predict a decrease in variability in energy consumption as time progresses.

## Time Series Energy Deep Learning 

Purpose:

The purpose of this notebook is to demonstrate the application of deep learning models for time series analysis and forecasting. Specifically, it aims to predict the 'Appliances' variable 24 hours into the future using various deep learning architectures. The notebook also includes exploratory data analysis and feature engineering steps to better understand the dataset and improve the model's performance. 

General Structure :

The notebook is structured as follows: 

1. Importing Libraries and Data: This section includes the necessary libraries and imports the energy dataset (`energydata_complete.csv`). 

2. Exploratory Data Analysis: Initial trends of the dataset columns are visualized using line plots. 

3. Feature Engineering: Additional features are created based on daily seasonality to enhance the model's understanding of the data. 

4. Splitting the Data: The dataset is split into training, validation, and test sets. 

5. Standardizing the Data: Feature scaling is performed to standardize the data and improve model convergence. 

6. Data Windowing: A custom `WindowGenerator` class is implemented to create data windows for the deep learning models. 

7. Deep Learning Models: Various deep learning models, including a Repeat Baseline, Linear Model, Dense Model, and Convolution Model, are defined and trained using the data windows. 

8. Model Evaluation and Visualization: The trained models are evaluated on the validation and test sets, and the results are visualized using line plots. 

Steps Taken and Why:

1. Importing Libraries and Data: The necessary libraries, including TensorFlow, matplotlib, numpy, pandas, statsmodels, seaborn, and sklearn, are imported. The energy dataset is loaded into a pandas DataFrame. 

2. Exploratory Data Analysis: Initial trends of the dataset columns are visualized using line plots. This step helps understand the data's distribution and identify any potential patterns or outliers. 

3. Feature Engineering: Two additional time features, 'Day sin' and 'Day cos,' are created based on the daily seasonality of the data. These features capture the cyclical patterns within a day and provide valuable information for the models. 

4. Splitting the Data: The dataset is split into training, validation, and test sets using the `train_test_split` function from sklearn. The split is performed based on a 70% - 15% - 15% ratio. 

5. Standardizing the Data: Feature scaling is performed using the mean and standard deviation of the training set. This step ensures that all features are on a similar scale and prevents any single feature from dominating the model's training process. 

6. Data Windowing: The `WindowGenerator` class is defined to create data windows for the deep learning models. It allows the models to learn from sequences of input data and predict future values. 

7. Deep Learning Models: Several deep learning models are implemented using TensorFlow's Keras API. The models include a Repeat Baseline, Linear Model, Dense Model, and Convolution Model. Each model is trained using the data windows created earlier. 

8. Model Evaluation and Visualization: The trained models are evaluated on the validation and test sets using mean squared error (MSE) and mean absolute error (MAE) as metrics. The results are stored in dictionaries for easy comparison. Additionally, the predictions of each model are plotted against the true values to visualize their performance. 

## SARIMAX Statistical Technique

Purpose:

The purpose of this project is to develop a SARIMAX model that can forecast the energy levels of certain appliances 24 hours into the future. The code in this notebook performs the data preprocessing, exploratory data analysis, statistical modeling, and visualization of the results. 

General Structure:

The notebook is structured as follows: 

1. Importing Libraries and Data: This section imports the necessary libraries and reads the dataset from a CSV file using pandas. 

2. Exploratory Data Analysis: This section explores the dataset by visualizing the initial trends of each column and generating descriptive statistics. 

3. Feature Engineering: This section checks for monthly and/or daily seasonality in the data. 

4. Statistical Model SARIMAX: This section applies SARIMAX modeling to forecast the energy consumption. It starts by making the data stationary by taking the difference of the dependent variable. The ad fuller test is performed to ensure stationarity. Auto correlation and partial auto correlation functions are plotted to determine the parameters for the SARIMAX model. The best parameters are selected using the optimize_SARIMAX function. The SARIMAX model is fitted with the selected parameters. The model diagnostics are plotted to check the residuals. 

5. Making Future Predictions: This section makes future predictions using the SARIMAX model. The predicted values and confidence intervals are obtained using the get_prediction function. The real values, predicted values, and confidence intervals are plotted to visualize the results. 

6. Predictions on Test Set and Performance Calculation: This section calculates the mean squared error (MSE) as a measure of model performance. 

Steps Taken and Why:

1. Importing Libraries and Data. The necessary libraries for data manipulation, visualization, and modeling are imported. The dataset is loaded from a CSV file into a pandas DataFrame. 

2. Exploratory Data Analysis. Initial trends of each column are visualized using line plots. Descriptive statistics are generated to gain insights into the data. 

3. Feature Engineering. The data is checked for monthly and/or daily seasonality using Fourier Transform. The frequency domain analysis is performed to identify periodic patterns. 

4. Statistical Models - SARIMAX. SARIMAX modeling is used to forecast the energy consumption. 

5. Making Future Predictions. Future predictions are made using the fitted SARIMAX model. 

6. Predictions on Test Set and Performance Calculation. The model's performance is evaluated by calculating the mean squared error (MSE) between the predicted and actual values. 

## Facebook Prophet Model

Purpose:

The purpose of this notebook is to demonstrate the application of the Prophet model for time series analysis and forecasting. Specifically, it aims to predict the 'Appliances' variable 24 hours into the future using the Prophet model. The notebook also includes exploratory data analysis and hyperparameter tuning steps to optimize the model's performance.

General Structure:

The notebook is structured as follows:

1. Importing Libraries and Data: This section includes the necessary libraries and imports the energy dataset (energydata_complete.csv).

2. Exploratory Data Analysis: The dataset is analyzed to gain insights into the data's distribution and patterns.

3. Prophet Model: The Prophet model is implemented for time series forecasting. The dataset is prepared by selecting the relevant columns and renaming them appropriately. The model is trained, and future forecasts are made.

4. Hyperparameter Tuning: The model's hyperparameters, such as changepoint_prior_scale and seasonality_prior_scale, are tuned using a grid search approach. The performance of each combination of hyperparameters is evaluated, and the best parameters are selected based on the root mean squared error (RMSE).

5. Making Future Forecasts: The trained model is used to make future forecasts by creating a dataframe with the desired time range. The forecasted values, along with the upper and lower bounds, are obtained.

6. Visualization: The forecasts and components of the time series (trend, yearly seasonality, and weekly seasonality) are visualized using line plots and interactive plots.

7. Performance Evaluation: The model's performance is evaluated using cross-validation. The mean absolute percentage error (MAPE) metric is used to assess the accuracy of the forecasts.

Steps Taken and Why:

1. Importing Libraries and Data: The necessary libraries, including TensorFlow, matplotlib, numpy, pandas, statsmodels, seaborn, and sklearn, are imported. The energy dataset is loaded into a pandas DataFrame.

2. Exploratory Data Analysis: The dataset is described using summary statistics, and irrelevant columns are dropped. This step helps understand the data's distribution and identify any potential issues or outliers.

3. Prophet Model: The Prophet library is installed and imported. The dataset is prepared by selecting the 'date' and 'Appliances' columns and renaming them to 'ds' and 'y', respectively. This step ensures that the data is in the required format for the Prophet model. The model is trained on the prepared data.

4. Hyperparameter Tuning: A grid search is performed to find the optimal values for the model's hyperparameters. The 'changepoint_prior_scale' and 'seasonality_prior_scale' parameters are varied, and the RMSE is calculated for each combination. The best hyperparameters are selected based on the lowest RMSE.

5. Making Future Forecasts: A future dataframe is created to specify the time range for which forecasts are desired. The trained model is used to predict the values for the future dataframe.

6. Visualization: The forecasts and components of the time series are plotted using matplotlib and Plotly. These plots help visualize the trends, seasonality, and uncertainty of the predictions.

7. Performance Evaluation: Cross-validation is performed to evaluate the model's performance. The MAPE metric is calculated to measure the accuracy of the forecasts. The results are visualized using line plots.


This README provides an overview of the energy consumption prediction project, including the data, tools, and skills required, as well as the results achieved using different time series analysis techniques. 
