# Importing libraries
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Importing csv file into dataframe
df = pd.read_csv("Consumo_cerveja.csv")

# Finding Number of rows and columns
print("Dataset contains {} rows and {} columns".format(df.shape[0], df.shape[1]))

# Looking first 5 rows of the dataset
print(df.head())

# Changing column names from portuguese to english
df.columns = ["Date", "Temp_Median", "Temp_Min", "Temp_Max", "Precipitation", "Weekend", "Consumption_Litres"]

# Changing commas to period(.)
for i in df.columns[1:5]:
    df[i] = df[i].str.replace(",", ".")

# Changing datatype of columns
for i in df.columns[1:5]:
    df[i] = df[i].astype("float")

# Looking first five rows after changes
print(df.head())

# Finding number of missing values
print(df.isnull().sum())

# Dropping null values
df = df.dropna()

# Converting the datatype of target column to float64
df['Consumption_Litres'] = pd.to_numeric(df['Consumption_Litres'])

# Visualisation of consumption in weekends and weekdays
weekdays = sum(df[df.Weekend == 0]['Consumption_Litres'])
weekend = sum(df[df.Weekend == 1]['Consumption_Litres'])

sb.countplot(df.Weekend)
plt.title("Count Plot of the Beer Consumption by weekend")
plt.show()

# Changing Date from string to date time
df['Date'] = pd.to_datetime(df['Date'])

# Extracting day and month from the datetime
df['Months'] = df['Date'].apply(lambda x: x.strftime('%B'))
df['Day'] = df['Date'].apply(lambda x: x.strftime('%A'))

# Visualizing consumption of beer by days of the week
sb.set_theme(style="whitegrid")
sb.boxplot(x="Day", y="Consumption_Litres", orient='v', data=df)
plt.title('Beer Consumption by day of the week')
plt.show()

# Visualizing consumption of beer by months of the year
sb.boxplot(x="Months", y="Consumption_Litres", orient='v', data=df)
plt.title('Beer Consumption by months of the year')
plt.show()

# Visualizing temperature throughout the year
df["Date"] = df["Date"].astype("datetime64[ns]")
plt.plot(df["Date"], df["Temp_Median"], label="Median Temperature")
plt.plot(df["Date"], df["Temp_Min"], label="Minimum Temperature")
plt.plot(df["Date"], df["Temp_Max"], label="Maximum Temperature")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.legend()
plt.show()

# Creating map for seasons according to temperature
seasons_map = {
    'January': 'Summer',
    'February': 'Summer',
    'March': 'Summer',
    'April': 'Autumn',
    'May': 'Autumn',
    'June': 'Winter',
    'July': 'Winter',
    'August': 'Winter',
    'September': 'Winter',
    'October': 'Spring',
    'November': 'Spring',
    'December': 'Summer'
}

df['Season'] = df['Months'].apply(lambda x: seasons_map[x])

# Visualizing consumption of beer by seasons
sb.boxplot(x="Season", y="Consumption_Litres", orient='v', data=df)
plt.title('Beer Consumption by Seasons of the year')
plt.show()

# Converting non-numerical labels to numerical labels
label_encoder = LabelEncoder()
df['Months'] = label_encoder.fit_transform(df['Months'])
df['Day'] = label_encoder.fit_transform(df['Day'])
df['Season'] = label_encoder.fit_transform(df['Season'])

# Converting date to float for giving it to learn the model
df["Date"] = df["Date"].values.astype("float64")

# Dropping target column from the dataset
X = df.drop('Consumption_Litres', axis=1)
y = df['Consumption_Litres']

# Splitting dataset in 80-20 (Train-Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Fitting the model
model = GradientBoostingRegressor().fit(X_train, y_train)

# Predicting the values for target feature
y_pred = model.predict(X_test)

# Creating output file
output = pd.DataFrame({'Date': X_test["Date"].values.astype("datetime64[ns]"), 'Beer_consumption': y_pred})
output.to_csv('GB_Output.csv', index=False)

# Finding error in the prediction
mae = mean_absolute_error(y_pred, y_test)
print("Mean absolute error using Gradient Boosting = ", mae)

# Actual vs Prediction Visualization
y_test_df = pd.DataFrame(data=y_test, columns=["Consumption_Litres"])
y_pred_df = pd.DataFrame(data=y_pred, columns=["Consumption_Litres"])

plt.plot(X_test["Date"].astype("datetime64[ns]"), y_test_df["Consumption_Litres"], label="Actual")
plt.plot(X_test["Date"].astype("datetime64[ns]"), y_pred_df["Consumption_Litres"], label="Predicted")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.title("Actual vs Predicted Beer Consumption")
plt.legend()
plt.show()
