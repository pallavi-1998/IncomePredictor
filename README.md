# Census Income Prediction
**The project aims to employ several supervised techniques to accurately predict individuals' income.** 
The importance of this project lies in, for example, helping non-profit organizations evaluate their much-needed donation requests from different individuals.

## Dataset
**The dataset that will be used is the Census income dataset**, which was extracted from the [machine learning repository (UCI)](https://archive.ics.uci.edu/ml/datasets/census+income), which contains about 32561 rows and 15 columns. The target variable in the data set is income level, which shows whether a person earns more than 50,000 per year or not, based on 14 features containing information on age, education, education-num,  gender, native-country, marital status, final weight, occupation, work classification, gender, race, hours-per-week, capital loss, and capital gain.

So, the target variable (income) will be represented by **binary classes**. **the class 0** for people having income less than or equal to 50k $ per year (<=50k $), and **the class 1** for people having income more than 50k $ per year (>50k $).

## Requirements
This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

## Environments
`CensusIncomePrediction/` (that contains this README) and run the following commands:

```bash
conda create -p venv python==3.9
```  
```bash
conda activate venv/
```
