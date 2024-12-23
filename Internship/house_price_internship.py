# -*- coding: utf-8 -*-
"""House_price_Internship.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1n96pJrdmtNLfNDxU-bVIx8gB_JNw4TaC
"""

#  Importing Necessary Libraraies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error ,r2_score

#Importing the Data
df=pd.read_csv("/content/Indian House Prices.csv")

df

"""### Lets Analyze the Dataset First"""

df.head()

df.tail()

df.shape

df.info()

df.describe()

df["City"].value_counts().sort_values()

"""### Now lets clean our Data"""

df.columns

len(df.columns)

"""#### -Remove not usefull columns"""

DF = df[['City', 'Location', 'Area','Gasconnection','PowerBackup', 'No. of Bedrooms', 'CarParking','School', 'AC','Hospital', 'Wifi', 'LiftAvailable', '24X7Security', 'Price']]
DF.head()

DF.columns

len(DF.columns)

"""#### -Columns reduces from 44 to 10
###
### Rename some Columns
"""

DF.rename(columns={'Area':'total_sqft', 'Location':'Area', 'No. of Bedrooms':'BHK', 'CarParking':'Parking', 'LiftAvailable':'Lift', '24X7Security':'Security'}, inplace=True)
DF.head()

"""### Adding an extra column for price per square feet."""

DF['Price_per_sqft'] = (DF['Price']*100000/DF['total_sqft']).round()
DF

DF.info()

DF["School"].value_counts()

DF["Hospital"].value_counts()

DF["Gasconnection"].value_counts()

DF["City"].value_counts()

#Checking Null Values
DF.isnull().sum()

DF["Area"]

plt.figure(figsize=(12, 8))
sns.histplot(data=DF, x='City', hue='BHK', multiple='dodge', shrink=.9)
plt.title('Prices distribution')
plt.show()

DF.sort_values(by=['Price'], ascending=False).head(20)

plt.figure(figsize=(12,8))
sns.countplot(x='BHK', data=DF, palette="Set3")
# Add labels and title
plt.xlabel("No of bedroom")
plt.ylabel("Count")
plt.title("BHK (Bedroom,Hall,Kitchen)")
#Show plot
for p in plt.gca().patches:
    plt.gca().annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+0.01))
plt.show()

plt.figure(figsize=(12,8))
sns.countplot(x='City', data=DF, palette="Set3")
# Add labels and title
plt.xlabel("No of City")
plt.ylabel("Count")
plt.title("City")
#Show plot
for p in plt.gca().patches:
    plt.gca().annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+0.01))
plt.show()

# Comparision of the prices for Floor in house
plt.figure(figsize=(15,8))
sns.boxplot(x="BHK",y="Price",data=DF,palette="Set3")
plt.xlabel("number_of_Bedrooms")
plt.ylabel("price")
plt.title("Comparision of the prices for different floors")

numerical_columns = DF.select_dtypes(include=['number']).columns.tolist()

# Print the numerical columns
print(numerical_columns)

corr = DF[numerical_columns].corr()
corr

plt.figure(figsize=(12,12))
sns.heatmap(corr, annot=True, cmap='coolwarm')

"""# Description of cities with price Range"""

city = 'Mumbai', 'Delhi', 'Chennai', 'Banglore', 'Hyderabad', 'Kolkata'
for c in city:
    df1 = DF[(DF['City']==c)]
    plt.figure(figsize=(10,8))
    sns.violinplot(df1['Price_per_sqft'], color='y')
    plt.title(f'Data distribution of Price per square-foot for {c} before Outlier removal', fontsize=16)

def remove_city_outliers(DF):
    data_new = pd.DataFrame()
    for key, subdata in DF.groupby('City'):
        mean = np.mean(subdata.Price_per_sqft)
        std = np.std(subdata.Price_per_sqft)
        new = subdata[(subdata.Price_per_sqft>(mean-std)) & (subdata.Price_per_sqft<=(mean+std))]
        data_new = pd.concat([data_new,new], ignore_index=True)
    return data_new
Df = remove_city_outliers(DF)
Df.shape

city = 'Mumbai', 'Delhi', 'Chennai', 'Banglore', 'Hyderabad', 'Kolkata'
for c in city:
    df1 = Df[(Df['City']==c)]
    plt.figure(figsize=(10,8))
    sns.violinplot(df1['Price_per_sqft'], color='orange')
    plt.title(f'Data distribution of Price per square-foot for {c} after Outlier removal', fontsize=16)

""" # Checking All outliers"""

# Function for Checking Outliers
def outliers(x):
    y=x.select_dtypes(include=[int,float])
    for i in y:
        sns.boxplot(x=DF[i])
        plt.title(i)
        plt.show()

outliers(DF)

"""# Remove outliers"""

DF.columns

def remove_outliers(DF, column):
    q1 = DF[column].quantile(0.25)
    q3 = DF[column].quantile(0.75)
    IQR = q3 - q1
    low = q1 - (1.5 * IQR)
    high = q3 + (1.5 * IQR)

    # Clip the values to the lower and upper bounds
    DF[column] = np.clip(DF[column], low, high)

# As these columns consisted with most outliers
remove_outliers(DF,'Price_per_sqft')
remove_outliers(DF,'Price')
remove_outliers(DF,'total_sqft')

"""# Modelling"""

DF1=DF.copy()

DF1.info()

y=DF1["Price"]
x=DF1.drop(columns=["Price","City","Area"])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=88)

"""# Defining_Models"""

models={"Linear Regression":LinearRegression(),
        "Decision Tree Regressor":DecisionTreeRegressor(),
        "Random Forest Regression":RandomForestRegressor()}

MSE_Results={}
for name,model in models.items():
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    MSE=mean_squared_error(y_test,y_pred)
    MSE_Results[name]=MSE

## PRINT (MSE) RESULTS
for name,mse in MSE_Results.items():
    print(f"{name}: MSE = {mse}")

print("++++++++++++++++++++++++++++++")

"""### Results dictionary"""

from sklearn.metrics import mean_squared_error

results = {}

# Iterate through models
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R-squared": r2}

# Print results
for name, metrics in results.items():
    print(f"{name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print()

"""### Convert results dictionary to DataFrame"""

results_df = pd.DataFrame.from_dict(results, orient='index')

# Print the DataFrame
print(results_df)

"""## -> BEST MODEL"""

## BEST MODEL
BEST_model= min(MSE_Results,key=MSE_Results.get)
print("--->> BEST MODEL IS ---> > > > ",{BEST_model})

