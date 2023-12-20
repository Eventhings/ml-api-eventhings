# Eventh!ngs Machine Learning Repository
<p align="center">
   <img src="https://github.com/Eventhings/backend-eventhings/assets/28957554/583ae431-25da-4796-8689-7fb1aba2a11d" width="500px"/>
</p>
In the ever-evolving landscape of event planning, Eventh!ngs emerges as a revolutionary solution, meticulously designed to simplify the intricate process of organizing events. The project’s Executive Summary encapsulates its core essence — a potent fusion of advanced technologies and user-centric design to redefine the event organization paradigm.

This repository is for the Machine Learning services used for Eventh!ngs Product Capstone Project

## Stack 
- Python v3.10
- Numpy (Numerical and Mathematical Operations)
- Pandas (Data Manipulation and Analysis)
- PostgreSQL (SQL Database)
- Sklearn (Model Metric Performance)
- Tensorflow (Recommendation System Model 1)
- FuzzyWuzzy (Recommendation System Model 2)
- NLTK (Natural Language Processing)
- Pickle (Save Model in .h5 Format)
- FastAPI (Deploy Models as APIs)

## Features
- [x] Content Based Recommendation System Model
- [x] Collaborative Filtering Recommendation System Model
- [x] Sentiment Analysis Model 

## Setup 
### Pre-requisite
- Python 3.10
### Initial Setup
1. Git clone this project using
```
git clone https://github.com/Eventhings/ml-api-eventhings
```
2. Install all needed dependencies using
```
pip install ./requirement.txt
```
3. Prepare all credentials needed (PostgreSQL) like provided on the `.env.example` file. The PostgreSQL database connected will be the same as the [Backend Services](https://github.com/Eventhings/backend-eventhings)
4. Make sure that your PostgreSQL Database is running
5. Start the server using 
```
uvicorn main:app
```
7. Enjoy the models as an APIs
   
## API Documentation
Postman Collection: https://documenter.getpostman.com/view/18445120/2s9Ykkgiqj
