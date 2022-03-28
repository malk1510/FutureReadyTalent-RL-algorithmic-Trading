# algorithmic Trading using Epsilon-Greedy Reinforcement Learning algorithm

## Problem Statement
Today, trading on the stock market and getting real-time feeds to check updates on any given stock has become faster than ever. Especially in the generation of High Frequency Trading using C to improve trading algorithms by nanoseconds, I wish to use Deep Reinforcement Learning to produce a faster algorithm for a fast and accurate algorithmic trading strategy on a given stock, and also attempt to implement a Garch-arima hybrid algorithm for time series forecasting of the given stock.

## Project Description
This project involves utilizing azure's ai and Machine Learning service to first prepare and test out common Reinforcement Learning algorithms such as the Sarsa algorithm, the TD(0) algorithm, DQNs and the actor-critic algorithm. I trained the epsilon-greedy RL algorithm using the datasets of stock values given on Yahoo Finance. Using this data, we could further predict the value of the stock in the future using time forecasting, and hence, predict whether or not a trade should be conducted at the given point of time or not. This is presented using a state-of-the-art data-based web application made using Streamlit on Python. This application will be deployed using the azure app service and web application service.

## Primary Azure Technology
AI + Machine Learning, App Service, Web Apps,

## Links
### Video Link
https://drive.google.com/file/d/1LUqDAh8Zkp25fIbitdFiZuV3XJNa8u6C/view?usp=sharing
### (Defunct) Site Link
https://rl-stock-app.azurewebsites.net

### NOTE: The link for the site will not work due to the expiration of my azure subscription. So, please find the implementation of the ML Notebooks and the Website on the Video Link given above.

## REFERENCE
To implement the code locally, simply type:

streamlit run main.py

into the terminal.
