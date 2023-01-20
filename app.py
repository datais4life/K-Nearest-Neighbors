# Import the standard libraries.

# Import and configure Pandas.
import pandas as pd
pd.set_option('display.precision',3)
pd.set_option('display.max_columns',9)
pd.set_option('display.width', None)

# Import and configure Scientific Computing.
import numpy as np
import scipy.stats as stats

# Import and configure plotting packages.
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rc("font", size=14)

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# Warnings
import warnings

# Import and configure SciKit Learn.
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics, tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

# Import Streamlit packages.
import streamlit as st
import altair as alt

# Setting Streamlit options
st.set_option('deprecation.showPyplotGlobalUse', False)

# Introduction
st.title("Telecom Churn Data: KNN Classification Model")

st.header("Overview")
st.write("Customer churn, or when a customer discontinues services with a company, is a problem that every business in the service industry deals with. This analysis will dive deeper into a data set for a telecommunications company, with the objective of understanding which customers are at a higher risk of churning from their company.")
st.write("The data set contains 10,000 customer records with 50 attributes used to describe them. The question of whether a new customer can be classified using certain data points in the data set will be analyzed. The method of analysis chosen is the K-Nearest Neighbors (KNN) machine learning algorithm. This analysis will classify a new control customer as either at risk or not of churning based on given metrics. For this, the features of Monthly Charge and Tenure will be used to build a model to classify our new customer and test the model’s accuracy.")

# Debug function
def log(msg, filename="streamlit_log.txt"):
    with open(filename, "a") as file:
        file.write(f"\n{msg}")

st.header("What is KNN Analysis?")
st.write("Classification analysis classifies a new data point based on its similarity to neighboring data points of attributes. Using the data points of other features in a data set, the new value can be categorized as a part of group of similar values. K-Nearest Neighbors (KNN) is a supervised machine learning algorithm that is perfect for this analysis. KNN is a widely used classification method used to calculate the distance between your target point and similar data points in the model. This method uses a straight-line or Euclidean distance to solve the problem.")

# Reading the data file
data = pd.read_csv('/app/original_files/churn_data/churn_clean.csv')

# Create a data frame using only the variables for this analysis.
clean = data[['Churn', 'Tenure', 'MonthlyCharge']].copy()

st.header("Telecom Churn Data")
st.write("The csv file is read and the variables needed for this analysis are separated and displayed in a table alonmg with the first five rows of data. This analysis will use the Tenure, Monthly Charge, and Churn attributes. The categorical variable of Churn will be changed from an object type with yes/no values to a Boolean type with true/false values. This will treat the variable as numeric for the model analysis. When comparing the target variable to the explanatory variables, the statistics can be seen.")

# Displaying the data frame.
st.table(data=clean.head())

# Data Exploration of raw data
#data.head()
#data.columns
#data.shape

# Data Cleaning of analysis data.
#clean.head()
# Check for null values.
#clean.isnull().sum()
# Check for duplicated rows.
#clean.duplicated().any()
# Check for outlying values in the numerical data using z-score.
clean_numerical = clean.select_dtypes(exclude = 'object')
z_scores = stats.zscore(clean_numerical)
abs_z_scores = np.abs(z_scores)
filtered_scores = (abs_z_scores < 3).all(axis = 1)
clean_wo_outliers = clean[filtered_scores]
#clean_wo_outliers.shape
# Rename the cleaned data set.
cleaned = clean_wo_outliers.copy()
#cleaned.head()

st.header("Comparing the Variables")
st.write("The target variable for this analysis is the churn feature. This variable is categorical and has yes/no values. Of the 10,000 customer records, 7,350 have not churned and 2,650 have left the company. The explanatory features that will be used are tenure and monthly charge. Both explanatory variables are continuous numerical data types.")

st.write("The average customer who left the company was paying \$199.30 monthly and had only 13 months of service. While those who have stayed only pay $163.01 on average and have been with the company for 42 months. This shows that the customers who are paying more are leaving early on into their service.")

st.write("The stacked bar chart displays the total monthly charge equating to the revenue broken down by the churn variable. The tenure is shown in the color variation to show the length of time the customers stayed with the provider.")

# Data Exploration 
# Variable statistics of the numerical data.
#cleaned.describe()
# Show the variables to be analyzed.
#for c in clean.loc[:, clean.columns]:
    #if clean.dtypes[c] == "object":
        #print('\n{} is categorical: {}.'.format(c,clean[c].unique()))
    #else:
        #print('\n{} is numerical:'.format(c ))
        #print('\trange = {} - {}'.format(clean[c].min(),clean[c].max()))
        #print('\tmean = {:.2f} +/- {:.2f}'.format(clean[c].mean(), clean[c].std()))

# Explore the churn variable. 
print(clean['Churn'].value_counts())
sns.countplot(x='Churn', data=clean, palette='hls')
plt.show()



# Visualizing the churn variable in a bar chart.
st.bar_chart(data=cleaned, x='Churn')

# Calculate the numerical mean data compared to churn.
#cleaned.groupby('Churn').mean().round(2).T

# Data Wrangling the target variable
# Convert churn values from yes/no to true/false.
target = 'Churn'
cleaned[target] = cleaned[target].replace({"No": False, "Yes": True})
cleaned[target] = cleaned[target].astype('bool')

# Scale the numerical variables.
features = ['MonthlyCharge', 'Tenure']
for c in features:
    cleaned['z'+c] = (cleaned[c] - cleaned[c].mean()) / cleaned[c].std()

# Show the clean data set for analysis.
def describeData(data):
    for idx, c in enumerate(data.columns):
        if data.dtypes[c] in ('float', 'int', 'int64'):
            print('\n{}. {} is numerical (CONTINUOUS) - type: {}.'.format(idx+1, c, data.dtypes[c]))
            if data.dtypes[c] in ('int', 'int64'):
                numbers = data[c].to_numpy()
                print('  Unique: {}'.format(get_unique_numbers(numbers)))
            if data.dtypes[c] in ('float', 'float64'):
                print('  Min: {:.3f}  Max: {:.3f}  Std: {:.3f}'.format(data[c].min(), data[c].max(),data[c]
                                                                       .std()))
        elif data.dtypes[c] == bool:
            print('\n{}. {} is boolean (BINARY): {}.'.format(idx+1,c,data[c].unique()))
        else:
            print('\n{}. {} is categorical (CATEGORICAL): {}.'.format(idx+1,c,data[c].unique()))
            
#describeData(data = cleaned)

# Splitting the data into training and testing groups.

# Define the primary feature and target data.
target= 'Churn' # target data
X = cleaned.loc[:, cleaned.columns != target]
y = cleaned.loc[:, cleaned.columns == target]

# Train / Test split the raw data.
tts = train_test_split(X, y, test_size=0.3, random_state=13)
(X_train, X_test, y_train, y_test)=tts
print('X_train: {}'.format(X_train.shape))
print('y_train: {}'.format(y_train.shape))
print('X_test: {}'.format(X_test.shape))
print('y_test: {}'.format(y_test.shape))

# Label the training data.
trainData = X_train.merge(y_train, 
        left_index=True, right_index=True)

# Label the test data.
testData = X_test.merge(y_test, 
        left_index=True, right_index=True)

# Create the new customer that will be used for the analysis.
newCustomer = pd.DataFrame([{'Tenure': 1.0, 'MonthlyCharge': 175.0, 'zTenure': 0.0, 'zMonthlyCharge': 0.0}])

st.header("KNN Model Analysis")
st.write("The classification analysis is performed by first defining the target and primary features and then splitting the data into training and testing groups. The control customer for the analysis is created to test the model. The model is built and run with the training data only and then again using all the data. The model is analyzed through a confusion matrix and the area under the curve is evaluated through the receiver operating curve graph.")
st.write("The data frame is split into training and testing data groups. This allows the model to be built on the training data and the accuracy can be tested on the testing data. First, the primary features and target data are defined in variables X and y.The data is split into these groups using Scikit-learn’s train_test_split function, where 70% of the data is for training and the other 30% is reserved for testing.")
st.write("The KNN analysis will determine if a new customer we create as a control will classify them as churn true or false, meaning more or less likely to leave the provider. First, the new customer is defined. The new customer will have a tenure value of 1.0 and a monthly charge value of \$175.00.")
st.write("A scatter plot of monthly charge compared to tenure is displayed using the training data and Seaborn.")

# Scatter plot of Monthly Charge vs Tenure using the training set.
fig, ax = plt.subplots()
fig.set_size_inches(8, 5)
sns.scatterplot(x='MonthlyCharge',y='Tenure',
    palette=['darkorange','blue'], hue=target,
    data=y_train.merge(X_train, left_index=True, right_index=True))

# Displaying a scatter plot of Tenure and Monthly Charge using training data.
scatterplot = alt.Chart(trainData).mark_circle().encode(
    x='MonthlyCharge',
    y='Tenure',
    color='Churn',
).interactive()

st.altair_chart(scatterplot, use_container_width=True)

# Scatter plot using the testing data set.
xFeature = 'MonthlyCharge' 
yFeature = 'Tenure'
target = 'Churn'
neighbors = []

def plotDataset(ax, data, xFeature, yFeature, target, neighbors, showLabel=True, **kwargs):
    
    """plot a given dataset with the target data on a given axis
    """
    # Churn == True
    subset = data.loc[data[target]==True]
    ax.scatter(subset[xFeature], subset[yFeature], marker='o',
              label=str(target)+'=True' if showLabel else None, color='C1', **kwargs)
    
    # Churn == False
    subset = data.loc[data[target]==False]
    ax.scatter(subset[xFeature], subset[yFeature], marker='D',
              label=str(target)+'=False' if showLabel else None, color='C0', **kwargs)
    
    # labels
    if len(neighbors) > 0:
        for idx, row in data.iterrows():
            ax.annotate(str(idx), (row[xFeature], row[yFeature])) 

fig, ax = plt.subplots()
fig.set_size_inches(8, 5)
plotDataset(ax, trainData, xFeature, yFeature, target, neighbors)
plotDataset(ax, testData, xFeature, yFeature, target, neighbors, showLabel=False, facecolors='none')

# Plot the new customer as a star.
ax.scatter(newCustomer.MonthlyCharge, newCustomer.Tenure, marker='*',
          label='New customer', color='red', s=270)

title = 'Scatter Plot'
plt.title(title)
plt.xlabel(xFeature) 
plt.ylabel(yFeature)

# Configure the legend.
handles, labels = ax.get_legend_handles_labels()
patch = mpatches.Patch(color='grey', label='Manual Label')
handles.append(patch) 
plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)

# Add the new customer data text.
plt.gcf().text(0, -.4, newCustomer.to_string(), fontsize=14)

st.write("The KNN model is created using the KNeighborsClassifier function from Scikit-learn.")

# KNN Model
knn = KNeighborsClassifier()
knn.fit(X = X_train, y = y_train['Churn'])

model = '''knn = KNeighborsClassifier()
knn.fit(X = X_train, y = y_train['Churn'])'''
st.code(model)

st.write("Predicted and observed data is defined. Incorrectly predicted data is defined and counted. There are 523 incorrectly predicted data points in the model. The KNN score is calculated for the model using the score method. This is the accuracy score for the model created. Hyperparameter tuning is performed on the model to find the number of nearest neighbors that should be used in the model. Running this shows that when k is equal to seven, the accuracy of 82% is achieved.")

# Predicted Data
predicted = knn.predict(X=X_test)

# Observed Data
observed = y_test['Churn']

# Incorrectly predicted data.
wrong = [(p,e) for (p,e) in zip(predicted, observed) if p!=e]
len(wrong)

# Calculate the KNN score for the model.
knn.score(X_test, y_test)

# Run Hyperparameter Tuning on the model.
for k in range(1,40,2):
    kfold = KFold(n_splits=10, random_state=11, shuffle=True)
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(estimator=knn,
            X=X_train, y=y_train['Churn'], cv=kfold)
    print(f'k={k:<2}; mean accuracy={scores.mean():.2%};')

st.code('''# Calculate the KNN score for the model.
knn.score(X_test, y_test)

# Run Hyperparameter Tuning on the model.
for k in range(1,40,2):
    kfold = KFold(n_splits=10, random_state=11, shuffle=True)
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(estimator=knn,
            X=X_train, y=y_train['Churn'], cv=kfold)
    print(f'k={k:<2}; mean accuracy={scores.mean():.2%};')''')


# Scale the new customer data.
newCustomer['zMonthlyCharge'] = (newCustomer['MonthlyCharge'] - cleaned['MonthlyCharge'].mean() ) /cleaned['MonthlyCharge'].std()
newCustomer['zTenure'] =  (newCustomer['Tenure'] - cleaned['Tenure'].mean() ) / cleaned['Tenure'].std()
newCustomerNorm = newCustomer[['zTenure','zMonthlyCharge']]
#newCustomerNorm

# Use NearestNeighbors from SciKit-Learn to compute KNN.
#k=7 # Determined from the hyperparameter tuning.
#knn = NearestNeighbors(n_neighbors=k)
#knn.fit(trainData.iloc[:,3:5])
#distances, indices = knn.kneighbors(newCustomerNorm)
#training_neighbors = trainData.iloc[indices[0],:]

# Show the training neighbors.
#d = training_neighbors
#display(d)

# Re-train the model with all data.
k=7
knn = KNeighborsClassifier(n_neighbors=k).fit(X[['zTenure','zMonthlyCharge']], y['Churn'])
distances, indices = knn.kneighbors(newCustomerNorm)
#print('Churn prediction (k={}) for \n{} is \n{}'.format(k,newCustomer,knn.predict(newCustomerNorm)))

# going through the model and finding the k using accuracy score
df_neighbors = cleaned.iloc[indices[0],:]
neighbors = df_neighbors.index
neighbors = neighbors.to_list()
#print(df_neighbors)

# Function to construct file names for figures and tables.
def getFilename(title: str, caption: str,
            sect='XX', ftype = 'PNG',
            subfolder='figures') -> str:
    """
    Construct a filename for given figure or table
    Input:
      title:
      sect:
      caption:
      ftype:
      subfolder:
    """
    temp = subfolder + '/'  # subfolder for tables and figures, default is 'fig'
    temp += sect + '_'
    temp += subfolder[0:3] + " " +caption + '_' #
    temp += title
    temp += '.' + ftype
    return temp.replace(' ','_').upper()

# Scatter plot using the plotDataset function
xFeature = 'MonthlyCharge' 
yFeature = 'Tenure'
fig, ax = plt.subplots()
plotDataset(ax, trainData, xFeature, yFeature, target, neighbors)
plotDataset(ax, testData, xFeature, yFeature, target, neighbors, showLabel=False, facecolors='none')

# Plot the new customer as a star.
ax.scatter(newCustomer.MonthlyCharge, newCustomer.Tenure, marker='*',
          label='New customer', color='red', s=270)

# Highlight neighbors with red circles.
if len(neighbors) > 0:
    for n in neighbors:
        point = clean.iloc[n]
        ax.scatter(point.MonthlyCharge, point.Tenure, marker='o',
                color='red', s=300, facecolors='none')

title = 'Final Model Prediction with Neighbors'
plt.title(title)
plt.xlabel(xFeature) 
plt.ylabel(yFeature)

# Set axis limits centered around the new customer.
left = float(newCustomer.MonthlyCharge) - 4
right = float(newCustomer.MonthlyCharge) + 4
top = float(newCustomer.Tenure) - 4
bottom = float(newCustomer.Tenure) + 3
ax.set_xlim(160,180)
ax.set_ylim(.8,2)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)

# get file names          
f = getFilename(title, sect='d2', subfolder='figures', caption='4 3') 
plt.gcf().text(0, -.2, title, fontsize=14) 

# Loop through the neighbors and include neighbors as table data.
plt.gcf().text(0, -.7, df_neighbors.iloc[:, 0:3].to_string(), fontsize=14)      
#plt.show()

st.write("The Scikit-learn NearestNeighbors function is used to compute the KNN analysis on the training data. Setting the k to seven as determined in the hyperparameter tuning, yields the seven customer records that are closest to the new customer. These all indicate that the new customer should be classified as churn false. The model is re-trained using all the data and k as seven again. This displays the seven customer records which are closest to the new customer’s metrics. This data is plotted in a scatter plot, with the new customer indicated as a star.")

# Plot final prediction with neighbors
final_prediction = st.pyplot()

st.write("A confusion matrix is created to assess the sensitivity (portion of the positive class that was correctly classified) and specificity (portion of the negative class that was correctly classified) of the model. This matrix shows that of 3000 predictions, 2477 were correct making the accuracy of the model 82.6%. Metrics for the confusion matrix verify this accuracy rate. This also verifies that there is a 17% chance of error. When put in terms of the new customer created to test the model, there is an 83% chance of them having the value of churn = True and a 17% chance of them having the value of churn = False.")

# Create the confusion matrix.
confusion = confusion_matrix(y_true=observed, y_pred=predicted)

# Create the plot for confusion matrix.
fig, ax = plt.subplots()
ax = sns.heatmap(confusion, annot=True, cmap='nipy_spectral_r', fmt='d')
title = 'Confusion Matrix (Churn Prediction)'
plt.title(title)
ax.set_xlabel('Predicted');
ax.set_ylabel('Observed'); 
ax.xaxis.set_ticklabels(['Positive', 'Negative']);
ax.yaxis.set_ticklabels(['Positive', 'Negative']);

# Add a filename.
f = getFilename(title, sect='E1', subfolder='figures', caption='5 1') 
plt.gcf().text(0, -.1, title, fontsize=14) 

# Add measurements.
TN, FP, FN, TP = confusion_matrix(y_true=observed, y_pred=predicted).ravel()
P = TP + FP
N = TN + FN
ERR = (FP + FN) / (TP + TN + FN + FP) # Error rate
ACC = (TP + TN) / (TP + TN + FN + FP) # Accuracy
SN = TP / (TP + FN) # Sensitivity
SP = TN / (TN + FP) # Specificity
PREC = TP / (TP + FP) # Precision
FPR = FP / (TN + FP) # False Positive Rate
COR = TP + TN

plt.gcf().text(0, -.3, 'Error rate (ERR): ' + str(ERR.round(3)), fontsize=14) 
plt.gcf().text(0, -.4, 'Accuracy (ACC): ' + str(ACC.round(3)), fontsize=14) 
plt.gcf().text(0, -.5, 'Sensitivity (SN): ' + str(SN.round(3)), fontsize=14) 
plt.gcf().text(0, -.6, 'Specificity (SP): ' + str(SP.round(3)), fontsize=14) 
plt.gcf().text(0, -.7, 'Precision (PREC): ' + str(PREC.round(3)), fontsize=14) 
plt.gcf().text(0, -.8, 'False Positive Rate (FPR): ' + str(FPR.round(3)), fontsize=14)  
plt.gcf().text(0, -.9, 'Correct Predictions (COR): ' + str(COR.round(3)), fontsize=14) 

matrix = st.pyplot()

st.write("A Receiver Operating Curve (ROC) was used to evaluate the model’s accuracy in predicting outcomes. This technique plots the true positive rate (TPR) against the false positive rate (FPR) to separate the ‘signal’ from the ‘noise’. The Area Under the Curve (AUC) is a calculation used to measure the ability of the model to distinguish between the positive and negative classes. The AUC value ranges between 0 to 1, with the value of 1 meaning that there is a high chance of positive values being distinguished from negative values. When analyzing the graph of the ROC, the higher X-axis value present shows a higher number of false positives than true negatives in the model. On the other hand, a higher Y-axis value shows a larger number of true positives than false negatives in the model.")

# Calculate the False Positive Rate (FPR) and True Positive Rate (TPR) for all thresholds of the classification.
fpr, tpr, threshold = metrics.roc_curve(observed, predicted)
auc = metrics.auc(fpr, tpr)

# Method I: Plot
fig, ax = plt.subplots()
title = 'ROC-AUC'
plt.title(title)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (TPR)')
plt.xlabel('False Positive Rate (FPR)')
f = getFilename(title, sect='e1',
    subfolder='figures', caption='5 2') # getFilename
plt.gcf().text(0, -.2, title, fontsize=14) 
#plt.show()

area_curve = st.pyplot()

st.header("Data Summary and Implications")
st.write("The final model is run using both the training and testing data sets. The accuracy, results and limitations of the model are outlined.  Recommendations on the practical use of this model are given.")
st.write("The final model proved to maintain an accuracy of 82% using k as seven to determine the seven closest customers with similar data points to the control customer. This was verified using the AUC in the ROC graph and by calculating the KNN score.")
st.write("The final seven similar customer records to the control customer are displayed. Five out of seven of the records indicate a true churn value. This means that the control customer is at a high risk (71% chance) of leaving the provider. This could be due to the monthly payment being higher than the average calculated for the whole data set. The control data can be manipulated, and the analysis run another time using lower parameters to confirm this theory.")
st.write("A limitation to this analysis is the variable selection. In choosing tenure, a new customer will always have a low value for this attribute. For the KNN computation however, it is not advisable to test a parameter with such a low value. A better variable selection could be made to compare the customer’s bandwidth usage or amount of outage time they experience instead to produce a better model for new customers. This model, however, will prove useful in classifying existing customers as likely or unlikely to churn.")
st.write("Additional analysis is recommended using this model with other variables such as bandwidth or outage time experienced to see if new customers can be evaluated more accurately. Using this model to evaluate existing customers is advised to determine if their monthly charge is too high based on their tenure and propensity to churn. An existing customer flagged by the model could be offered a discount on their monthly service to decrease their risk of churning and increase their tenure.")