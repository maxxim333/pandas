#Importing all the necessary modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import scipy.stats as stats
from sklearn.model_selection import train_test_split #For splitting data in training and test
from sklearn.ensemble import RandomForestRegressor
from dateutil.parser import parse #Operations with dates
from sklearn.model_selection import RandomizedSearchCV




####################################### DATA WRANGLING #####################################################
#Read the file
df = pd.read_csv('C:/Users/Maksym/Desktop/Jahbs/berlin_intr/investments_VC.csv', encoding= 'unicode_escape')

#On my PC, I worked with a subset of data because of computational power limits
df=df.sample(n = 15000, random_state=42) 

#Dataset has "-" values that I want to be iterpret as "NaN" instead
df1=df.replace('-', df.replace(['-'], [None])) 

#The variable I want to try to predict is Total Funding in USD, so I will remove all the rows where funding_total_usd is unknown
df1 = df1[df1['funding_total_usd'].notna()]

#Make a copy of the dataframe (It will be used later for some comparisons)
df=df1

#For now, lets do some data inspecting
#See all the columns (there are a lot of them)
for col in df.columns: 
     print(col)

#We can visualize the distribution of individual variables. Here is the example of equity_crowdfunding
df[["equity_crowdfunding"]].plot(bins=4, kind="hist", density=1)
#plt.show()

#Some data is better visualized with a CDF (cumulative distribution function) instead of density function. A good example in this data is funding_rounds
df.funding_rounds.hist(bins=5, cumulative=True, density=1)
#plt.show()

#Distribution separated by other variable. For example, if we want to plot total funding VS funding round separatedly for each startups that were acquired, closed or still running
df[[ 'funding_total_usd','funding_rounds']].hist(by=df.status)
#plt.show()

#When preparing for machine learning analysis, we need to create dummy variables for categorical variables:
#In this data there will be a lot of them. For example, just by inspecting the first entries of "Market" variable, we can see there will be variety. Every different market will be a new variable with a value 1 or 0
print(df["market"].head())

#Code to create dummy variables of funding_rounds as an example
df_funding_rounds_dummy = pd.get_dummies(df["funding_rounds"], prefix='funding_rounds_')

#Once we have these variables, we drop the original variable from dataframe
df = df.drop('funding_rounds', axis=1)

#Add dummy variables
df = pd.concat([df, df_funding_rounds_dummy], axis=1)

#Lets see the shape of our data. It should have 242 rows (aprox) and 47 columns (exactly)
print("data-frame shape: ", df.shape)

#Inspect how many missing values do we have in total (in each of the columns)
print(df.isna().sum())

#Sometimes there are duplicate entries in the data. We can eliminate it as it makes a biased learning
df.drop_duplicates()

#Visualize the "venture" variable in a boxplot
sns.boxplot(x=df['venture'], color='lime')
plt.xlabel('Venture in USD', fontsize=14)
#plt.show()
#The graph is ruined by outliers. We can make another graph without them
sns.boxplot(x=df['venture'], color='lime', showfliers = False)
#plt.show()

#Describing a particular column (gives us mean, standard deviations and percentiles)
print(df['venture'].describe().apply(lambda x: format(x, 'f'))) #Here I suppressed the scientific notation

#For some applications, one needs to check data types of each column
print("data types: \n", df.dtypes)

#We might want to eleminate rows that have an outlier value for any particular variable(s). 
#For example, here I want to delete all the rows of all columns if that row has a value greater than 4 times the zeta score of that column. This method is good because it corrects for standart deviation of each column
df = df.select_dtypes(exclude=['object']) #exclude all non-numeric
print("shape during rejecting outliers: ", df.shape)
df2 = df[(np.abs(stats.zscore(df))<4).all(axis=1)]
print("shape after rejecting outliers: ", df2.shape)

#################################### Data processing and preparation for ML  ###############################
#For now, it doesnt work because the dates are in a wrong format (for example 12/04/2000 will not be interpreted as an integer or float by the regressor). 
# I want to see what will happen when I do only the strictly necessary data preparation and no tuning of the predictor


#Before, in this code I was using the variable name df1 for a my data. I will make it be "df" now
df=df1

#Set labels
labels = np.array(df['funding_total_usd'])


#I will delete the columns with dates (later I will do it with a better solution) and also the non-relevant columns
#Also, for the categorical columns, I will create dummy variables
df.columns = df.columns.str.strip() #Deletes all strings
df= df.drop(['funding_total_usd', "state_code",'permalink', "name", "homepage_url","founded_at",	"founded_month",	"founded_quarter",	"founded_year",	"first_funding_at",	"last_funding_at"], axis = 1)
df=pd.get_dummies(df, columns = ['status', 'region', 'market', 'city', 'category', 'country_code'])

#Visualize all columns in the new dataset to see if the columns were dropped correctly
print(df.columns.values)

#This just renames dataframe to "features"
features= df

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

#Take a look of the shape of data
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# Instantiate model with X decision trees
rf = RandomForestRegressor(n_estimators = 10, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions_1 = rf.predict(test_features)

#I decided to mesure the quality of prediction with a average of absolute values of errors in percentage (relative to the value itself), given by average of (abs((prediction - real_value)/real_value)*100)
#The following block of code defines a function that does two things: calculates and outputs the average of absolute errors and 
#Constructs a distribution function based on average and standart deviationn (assumes there is normal distribution)
def average_error(predictions):
     error_array=[] #here I will store the values of absolute errors
     error_array_not_absolute=[] #here I will store the values of errors

     for i in range(0,len(predictions),1):
          errors_not_absolute = int(round(predictions[i]))-int(test_labels[i])
          errors = abs(int(round(predictions[i]))-int(test_labels[i]))

          errors_not_absolute = (errors_not_absolute/int(test_labels[i]))
          errors = (errors/int(test_labels[i]))

          error_array_not_absolute.append(errors_not_absolute*100)
          error_array.append(errors*100)

          i=i+1

     mu = np.mean(error_array_not_absolute) #mean
     sigma=np.std(error_array_not_absolute, ddof=1) #standar deviation

     x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
     plt.plot(x, stats.norm.pdf(x, mu, sigma))
     plt.show()

     print("The average error is: " + str(np.mean(error_array)) + " percent")

#Lets see the average error of a "plain" regressor
#average_error(predictions_1)


######################################### IMPROVING MODEL #####################################################
#The predictive power can be improved by two ways: improving the data and tuning the predictor. Here, I focus on the first strategy
#First of all, in the previous prediction I deleted variables that might be valuable. Here, I will use recover one of them, namely founded_year that might be important
#Secondly, just for curiosity I will create a new variable called "time_until_first_fundings" which is defined as first funding year minus the founding year. Basically how much time passed between the creation of the company and the first influx of money
#For that, I will use dateutil module that allows doing operations directly with dates (eg: 12/01/2018 - 12/01/2017 = 365 days)

#Again, re-set the dataset
df=df1

#Calculate new variables timeuntilfirst_fundings
df['time_until_first_fundings'] = 0
for i in range(0, len(df['time_until_first_fundings'])):
    if "nan" not in str(df['founded_at'].values[i]):
        df['time_until_first_fundings'].values[i] = (abs(parse(str(df['first_funding_at'].values[i])) - parse(str(df['founded_at'].values[i]))).days)

#The variable 'founded_year was unusable because it contains unknown values. There are a lot of ways to deal with this problem, but here, I just substituted all the unknown values with the mean of the column
df['founded_year'].fillna((df['founded_year'].mean()), inplace=True)

#As before, the label that we want to predict is funding_total_usd
labels = np.array(df['funding_total_usd'])

#Drop unnecessary columns, create dummy variables for categoric variables, split train/test and train model. This time, I dont drop the founded year column. Everything else is the same
df.columns = df.columns.str.strip()
df= df.drop(['funding_total_usd', "state_code",'permalink', "name", "homepage_url","founded_at",	"founded_month",	"founded_quarter",	"first_funding_at",	"last_funding_at"], axis = 1)
df=pd.get_dummies(df, columns = ['status', 'region', 'market', 'city', 'category', 'country_code'])
features= df
feature_list = list(features.columns)
features = np.array(features) # Convert to numpy array
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
rf.fit(train_features, train_labels)

#Predict and evaluate predictor
predictions_2 = rf.predict(test_features)
#average_error(predictions_2)

#Some of the variables used might be more important than others. We can see the important variables like this
#Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
#Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = False)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

#Additionally, some variables might be redundant. Here I want to visually represent redundant variables by constructing a matrix plot
#Because of the dummy variables, there is too many columns so for visual simplicity, I will select a subset of variables to draw a matrix plot
df2 = df[['funding_rounds', 'founded_year', 'seed', 'venture' ,'equity_crowdfunding',
 'undisclosed', 'convertible_note', 'debt_financing', 'angel', 'grant',
 'private_equity', 'post_ipo_equity', 'post_ipo_debt',
 'product_crowdfunding' ,'time_until_first_fundings',
 'status_acquired' ,'status_closed' ,'status_operating']]
corrMatrix = df2.corr()
sns.set(font_scale=0.8)
sns.heatmap(corrMatrix,annot_kws={"size":5}, annot=True)
#plt.show()

#What will happen if I do exactly the same predictor as before but remove one of the variables from each pair of variables that has >95% correlation with another?
#The block of code below does exactly that. #Taken from cool resourse https://chrisalbon.com/#python
#The following two functions get the top n most correlated variables.
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df, 30))

# Create correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features 
df=df.drop(df[to_drop], axis=1)

# Train, split, predict...
features= df
feature_list = list(features.columns)
features = np.array(features)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
rf.fit(train_features, train_labels)
predictions_3 = rf.predict(test_features)

#Evaluate 
#average_error(predictions_3)

######################################################### HYPERPARAMETERS TUNING ################################################
# Now, I will try to improve the predictor by tuning the hyperparameters of the regressor using a randomized search for optimal parameters (that output the lowest error)

#The block of code below defines the grid of parameters
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 200, num = 50)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 500, num = 30)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
#Random state has to be the same so its comparable
random_state=[42]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'random_state': random_state}
#Random state has to be the same so its comparable
random_state=[42]

# Create the base model to tune
rf = RandomForestRegressor()

# Search across 100 different combinations, and use all available cores (using 3x-cross-validation)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 95, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_features, train_labels)
#Output the Best Params
rf_random.best_params_

#The function below estimates the average error of the model of input, given the test features and labels
def evaluate(model, test_features, test_labels):
    errors=[]
    predictions2 = model.predict(test_features)
    for i in range(0,len(predictions2),1):
            error = abs(int(round(predictions2[i]))-int(test_labels[i]))
            error = (error/int(test_labels[i]))
            errors.append(error*100)
            i=i+1
    print('Model Performance for ' + str(model))
    print('Average Error: {:0.4f} percent.'.format(np.mean(errors)))
    return " "

#Compare the base model (the one used until now) with the best model from the rando grid search
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)

#Launching average error function on all the possible ways of regressors tried
predictions_4=base_model.predict(test_features)
predictions_5=best_random.predict(test_features)

average_error(predictions_1) #Simple model with minimal data preparation
average_error(predictions_2) #Adding funding year and time until first funding variable
average_error(predictions_3) #Removing highly correlated variables
average_error(predictions_5) #Hyperparameters tuning

#Model Performance for RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
      #                max_depth=None, max_features='auto', max_leaf_nodes=None,
     #                 max_samples=None, min_impurity_decrease=0.0,
    #                  min_impurity_split=None, min_samples_leaf=1,
   #                   min_samples_split=2, min_weight_fraction_leaf=0.0,
  #                    n_estimators=10, n_jobs=None, oob_score=False,
 #                     random_state=42, verbose=0, warm_start=False)
#Average Error: 11.4450 percent.
#Model Performance for RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
      #                max_depth=466, max_features='auto', max_leaf_nodes=None,
     #                 max_samples=None, min_impurity_decrease=0.0,
    #                  min_impurity_split=None, min_samples_leaf=1,
   #                   min_samples_split=2, min_weight_fraction_leaf=0.0,
  #                    n_estimators=20, n_jobs=None, oob_score=False,
 #                     random_state=42, verbose=0, warm_start=False)
#verage Error: 9.9569 percent.
#The average error is: 11.549142304907223 percent
#The average error is: 12.650536984290637 percent
#The average error is: 11.444957930406002 percent
#The average error is: 11.444957930406002 percent
#The average error is: 9.956937750680716 percent
