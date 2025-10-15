# Natural Gas Price Detection Using Machine Learning Model

## Introduction
### Overview
Forecasting natural gas prices is an important tool that helps many people and organizations involved in the natural gas market. Accurate price predictions allow them to make smarter decisions, manage risks more effectively, balance supply and demand, and use resources more efficiently.

### Purposed
Accurately forecasting natural gas prices plays a key role in shaping effective energy policies and planning. It is also very important for economic planning, energy investments, and protecting the environment. The goal of this project is to develop data-driven machine learning models to predict natural gas prices.

## Literature Survey
### Existing Problem
Energy prices are hard to predict, so experts often look at energy futures markets to guess future prices. These markets trade contracts that set prices for natural gas to be delivered later — so they show what people expect prices to be in the future.
Different studies have found mixed results about how accurate these futures prices are:
Walls (1995): Found that natural gas futures generally predict future prices quite well (no big bias).
Herbert (1993): Found that futures prices are often too high compared to the actual prices later.
Chinn et al. (2005): Found that futures prices are usually good predictors, except for natural gas prices over a three-month period, but still slightly better than traditional forecasting models.

### Proposed Solution
Using the data set of prices provided from the 7th of January 1997 until 29th November 2025, we will be trying to predict the prices of natural gas by testing through various machine learning models and providing a real-time web based GUI to ask the user to enter the desired date to predict the rate of the natural gas on that particular day. This study builds upon the existing literature by investigating the accuracy of various forecast methods until the best fit horizon is reached.

## Theoretical Analysis
### Block Diagram
![Image](https://github.com/saidavali123/Natural-gas-price-prediction/blob/master/Natural-gas-price-prediction-master/Natural-Gas-Price-Prediction-System-main/Images/Technical%20Architecture.png)
### Software Designing
We used the following software tools and Python libraries:
Flask (for web app)
Pandas, NumPy (for data handling)
Scikit-learn (for machine learning)
Matplotlib, Seaborn (for data visualization)
SciPy, Pydotplus, Six (for calculations and graphing)
Jupyter, Spyder, Notebook (for development)
Gunicorn (for deployment)

Python packages used:
Flask==3.0.3
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.2
matplotlib==3.9.2
seaborn==0.13.2
scipy==1.13.1
pydotplus==2.0.2
six==1.16.0
ipython==8.27.0
gunicorn==23.0.0
jupyter==1.1.1
notebook==7.2.2
spyder==6.0.1

### Machine Learning Workflow
Data Preprocessing (handling missing values, removing outliers)
Feature Extraction (year, month, day)
Model Training and Testing (Decision Tree, Random Forest)
Model Evaluation (accuracy and performance comparison)
Web Application Development using Flask

## Experimental Investigations

The dataset was preprocessed and trained using Decision Tree and Random Forest regression models.
Decision Tree Regressor Accuracy: 97.4%
Random Forest Regressor Accuracy: 97.74% ✅
The Random Forest model was deployed in the web app because it achieved the best accuracy and stability.

## Flow Chart
![Image](https://github.com/saidavali123/Natural-gas-price-prediction/blob/master/Natural-gas-price-prediction-master/Natural-Gas-Price-Prediction-System-main/Images/Flow%20chart.png)

## Advantages and Disadvantages
| Prediction Model  | Advantages | Disadvantages |
| ------------- | ------------- | ------------- |
| TS models  | The model is relatively simple. It performs well in comparatively stable datasets. Provides good result in short and medium termed forecasts  | Hugely limited in long-term forecasting |
| Regression models  | A simple model, but the modeling speed is fast. Predicts reliably and can indicate the relationship and influence strength between independent variable and the dependent variable.  | In some cases, the choice and expression of factors are purely speculative and limited in application. |
| ANN based models  | It has a strong nonlinear fitting ability, simple self learning rules, and easy to perceive by a computer. It has strong robustness, memory ability, non linear mapping ability and strong self learning ability. | The interpretability of this model is poor and prone to overfitting |
| SVM based models  | It performs better in small sample problems with a basic algorithm and strong robustness | It is difficult to implant for large scale training samples. Sensitive to missing data, choice of parameters and kernel functions. |
| Decision Tree Based Models | The model has strong generalization ability,can be trained fast,and is not sensitive to missing data | If there is noise in the data it is prone to overfitting. |

## Application
Natural gas makes up about one-fourth of global energy demand and about one-third of the energy used in the United States. It is the second most important energy source after oil. Therefore, improving the prediction of natural gas demand is very valuable.
Accurate energy price forecasts are essential for guiding energy markets and helping both policymakers and market participants make informed decisions. However, predicting energy prices is difficult because they are influenced by many external factors.
Being able to forecast natural gas prices provides major benefits to different stakeholders and is a valuable tool in competitive energy markets. In recent years, machine learning methods have become increasingly popular for forecasting natural gas prices.


## Conclusion
It has always been a difficult task to predict the exact daily price of natural gas price. Many factors such as political events, general economic conditions, and traders’ expectations may have an influence on it. But here, based on the past and present traits, we were able to achieve up to 97% accuracy in predicting the price of any given date. Albeit, its impossible to predict unexpected scenarios such as acts of warfare or terrorism. But, the benefits of having reliable information of what the price of natural gas could be at any given time is paramount, it could make or break economies. And in this case, as this project points out data-driven machine learning models deserve all the attention it could ever garner and even more.

## Future Scope
The project has been built using 2 models of prediction namely the Decision Tree method and Random Forest method with the accuracy score of over 97% on both the models (97.4% on Decision Tree and 97.74% on Random Forest Method). By doing some further research and learning the accuracy can be uplifted upto 100% which would be an ideal prediction real- time application which would be much more helpful in the trading sector.

## Bibliography
- https://apsche.smartinternz.com/
- https://docs.anaconda.com/anaconda/packages/pkg-docs/
- https://iea.blob.core.windows.net/assets/ef262e8d-239f-4cfc-8f8c-4d75ac887a0f/IndiaGasMarketReport.pdf
- https://www.mdpi.com/1996-1073/15/10/3573
- https://www.iea.org/news/global-natural-gas-demand-growth-set-to-accelerate-in-2026-as-more-lng-supply-comes-to-market
   
## UI Output Screenshots
![Image](https://github.com/saidavali123/Natural-gas-price-prediction/blob/master/Natural-gas-price-prediction-master/Natural-Gas-Price-Prediction-System-main/Images/UI%20image.png)

## Demonstration Video


### Contributors
- [Shaik saidavali](https://github.com/saidavali123)
- [Shaik shameer]()
- [Shaik Nasreen]()
- [Shaik Shaziya Mehzabee]()

