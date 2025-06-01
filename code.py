import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plot
#import the data from customer.csv file
data=pd.read_csv('E:/Projects/KFiles/js work/python learn/Machine-Learning-Deep-Learning-in-Python-R/Data Files/1. ST Academy - Crash course and Regression files/House_Price.csv',header=0)#hwaders=0 coz headers are in 1st row

print(data.head())
#get the EDD of the houe_price
EDD=data.describe()
print(EDD)
#the whole information all the variables of the dataset is called the extended data dictionary
graph=sb.jointplot(x='n_hot_rooms',y='price',data=data)
plot.show()
#print graph for category variables
graph2=sb.countplot(x='airport',data=data)#note can't print both x and y variables
                                        #for horizonal graphs give valu in y,for vertical graphs give value in x,
plot.show()
graph_buster=sb.countplot(x='bus_ter',data=data)
plot.show()
#get teh information of the data
data.info()

#defining outliesr in data
#get the 99 percentile

percentile=np.percentile(data.price,[0])#return array containing length of 
up=percentile[0]
print(up)

#fetch teh data havinmg value above 99percentile of n_hot_rooms
result=data[(data.rainfall>up)]
print(result)

#limit the value of outliers in data
data.rainfall[(data.rainfall>3*up)]=3*up
new_result=data[(data.rainfall>up)]
print(new_result)

#check whether crime arte includes outliners or not given crime_rate depends upon price
#graph5=sb.jointplot(x='crime_rate',y='price',data=data)
#plot.show()
#transform the graph into log,sqroot,etc so that outliners can remove automatically

print(data.describe())


#missing values in python
#get the missing values
print(data.info())

#Ques=>disadvatantage of using EDD for finding missing variable
#ans=>data.describe() gives min, max values while data.info() ggives categorial results

#find missing variable hospital bed
data.n_hos_beds=data.n_hos_beds.fillna(data.n_hos_beds.mean())#in this step values which are 0 or NA ar efilled with the mean of hospital beds
print(data.info())

#ques=>find the mean of the missing variable data waterbody

#data.waterbody=data.waterbody.fillna(data.waterbody.mean()) 
#error occurred in above line coz  we cant add int to str


#plot scatterplot of crime rates and prices
graph=sb.scatterplot(x="crime_rate",y="price",data=data)
plot.show()
#method 2 more convinient
graph2=sb.jointplot(x="crime_rate",y="price",data=data)
plot.show()

#above curve seems like log curve 
#Ques=>transofrm and find linear relationship or convert into linear relationship of variables
data.crime_rate=np.log(1+data.crime_rate)
graph3=sb.jointplot(x="crime_rate",y="price",data=data)
plot.show()


#ques=> make a new variable named  meandistance which is equivalent to mean of distances of dist1,dist2,dist3,and dist4
data['meandist']=(data.dist1+data.dist2+data.dist3+data.dist4)/4


#delete the  collumns dist1,dist2,dist3,dist4,bus_ter from the data
del data['dist1'],data['dist2'],data['dist3'],data['dist4'],data['bus_ter']

#ceate dummy variables for category airport and waterbody
data=pd.get_dummies(data,dtype=int)#dtype=int for i=ensuring values must be 0 and 1 not True and False

print(data.info())
print(data.head())

#delete the unwanted dummmy variables
del data['airport_NO'],data['waterbody_Lake and River']
print(data.info())


#corelation analysis 
#ques get the correlation matrix of data
print(data.corr())
#note:ValueError: could not convert string to float: 'YES' means data cntains string value but value corelation matrix always contains float or int values

#check colrelation of price with other variables
#air_equal and parks have high corelational value  both for together and with respect to price so we need to remove any one 

del data['parks']
print(data.info())


#linear regression in python
#method-1
#task=1 import module statsmodel.api
import statsmodels.api as stats
#add constant to the equation
x=stats.add_constant(data['room_num'])
#create the model
lm=stats.OLS(data['price'],x).fit()#whats the diffrence between ols and wls?
                                #beta0=> price,beta1=>room_num
                                #p value nearly=0 means that significant relationship between variables

#get the summary of the model
print(lm.summary())

#method 2
from sklearn.linear_model import LinearRegression #LinearRegression is  library for ml
#define x and y variables for equation
y=data['price']
#cerate a 2d array for x {why?}
X=data[['room_num']]
lm2=LinearRegression()
#fit the variables in the model
lm2.fit(X,y)
#see the iontercept and coefficiants of teh model
print(lm2.coef_,lm2.intercept_)
#predict he values of  and y
prx=lm2.predict(X)# dot predict expects a 2d array as input therefor can't give y as input here as y is 1d array
print("predicted values of y are given below:")
print(prx)

graphlm=sb.jointplot(x=data['room_num'],y=data['price'],data=data,kind='reg')
plot.show()

#multiple regressionn model in python
#cerate multiple independednt variabeles and drop price variable
x_multi=data.drop("price",axis=1)#axis=1 for dropping collumns
                                #axis=0 for dropping rows
#create dependernt variable
y_multi=data['price']
#add constant to thw variable
x_multi_const=stats.add_constant(x_multi)
print(x_multi_const.head())

#create lm model for multiple regression
lm_multi=stats.OLS(y_multi,x_multi_const).fit()#why not written x_multi?
#veiw the model 
print(lm_multi.summary())

#note:formula for degree of freedom=n-p-1
#       lower the p valuemore significant  variable is determining  y
#     how to intrepert data=> 1)see the sign, if sign is positive=> increasing independent variable will increase dependent variable 
#                              2) if sign is -ve,increasing independent vriable will decrease dependent variable


#create the model using sklearn library
lm3=LinearRegression()
lm3.fit(x_multi,y_multi)
print(lm3.coef_,lm3.intercept_,end="\n")

#note: MSE=mean square error

#test and split your model
#
from sklearn.model_selection import train_test_split
#create 4 variables 
x_train,x_test,y_train,y_test=train_test_split(x_multi,y_multi,test_size=0.2,random_state=0)#we are training 80% of the dtaa and using rest 20%of the data for testing
                                                                                            #random state=0 ,get same sample eery time
#check the no. of rows and collums in your training set
print("shapes are :",x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#train your model
lm_a=LinearRegression()
lm_a.fit(x_train,y_train)
#predict the value 
y_test_a=lm_a.predict(x_test)
y_train_a=lm_a.predict(x_train)

#check the rsquare value for your testing data
from sklearn.metrics import r2_score
value=r2_score(y_test,y_test_a)#syntax=>  first test value and then predicted value is given
print(value)
#ques=> get the value for the training set also
value_train=r2_score(y_train,y_train_a)

#laso and briding
#standardize the data
from sklearn import preprocessing
#create the sclaer oblect for storing scaler information
scaler=preprocessing.StandardScaler().fit(x_train)
#transform train into train scale
x_train_s=scaler.transform(x_train)
x_test_s=scaler.transform(x_test)

#ridge regrewssion
from sklearn.linear_model import Ridge
#creating a ridge object
lm_r=Ridge(alpha=0.5)#alpha here refers to lambda
#fit this model
lm_r.fit(x_train_s,y_train)#here we are fitting our scaled variables
#find teh r-square value of this data
r2_score(y_test,lm_r.predict(x_test_s))#y-test is our original values, nd argument is predicted values of y on test variables

#train model for  multiple lambdaas or alphaas value
from sklearn.model_selection import validation_curve
#create array for multiple lambdaa values
param_range=np.logspace(-8,100)#check more about this fn
#run the iteration fir teh mdoel
trian_scores,test_scores=validation_curve(Ridge(),x_train_s,y_train,param_name="alpha",param_range=param_range,scoring='r2')#uses k-fold method
print(trian_scores)
print(test_scores)
#take the mean of these 3 values
train_mean=np.mean(trian_scores,axis=1)
print(train_mean)
#take mean value of test scores
test_mean=np.mean(test_scores,axis=1)
print(test_mean)
print(train_mean)
#get the highest value of model having highest R-square
max(test_mean)
"""
error portion
#draw the graph of your model
graph5=sb.jointplot(x=np.log(param_range),y=test_mean)
plot.show()
"""

#check the location of ma r-square value
location=np.where(test_mean==max(test_mean))
print(location)
print(param_range[4])
#create the best model 
lm_best=Ridge(alpha=param_range[4])
#fit this into training dataset
lm_best.fit(x_train_s,y_train)
#find the r square value on test data
r2_score(y_test,lm_best.predict(x_test_s))
#also find r square value for train data
r2_score(y_train,lm_best.predict(x_train_s))
#homework=> eecute lasso on your own


#LOGISTIC REGRESSION
#method-1 creating logistiic model using sklearn
#create your x variable
X=data[['price']]#in sklearn x variable must be 2-d
#NOTE: logistic regression model accepts a dicrete value in Y i.e in y variable the value must be binary
Y=data['airport_YES']
#import logistic from sklearn 
from sklearn.linear_model import LogisticRegression
#steps for logistic regression in sklearn
# 1=>creating classification object
#2=>fit our object using x and y variable
#3=>predict y variable

#step-1 code
clf_lrs=LogisticRegression()
#step-2 code
clf_lrs.fit(X,Y)
#predict and check value of coefficients
clf_lrs.coef_
clf_lrs.intercept_

#method-3 creating logistic model using stats model library
#drawback=>by deafult stats model does nt use a constant term i.e beta0 will be 0 by default

#add constant to model 
X_const=stats.add_constant(X)
#import the logistic regression in stats model
import statsmodels.discrete.discrete_model as sm 
#train your x and y variable
logistic=sm.Logit(Y,X_const).fit()
#see the summary iof the model
logistic.summary()

#creating a confusion matris in python
#predict the value of yhe model
clf_lrs.predict_proba(X)

y_pred=clf_lrs.predict(X)# by default takes boundary conditions as 0.5
                  #output will be in the form of binary/boll values in array according to the prediciton

#set the custom boundary conditions
y_pred_03=(clf_lrs.predict_proba(X)[:,1]>=0.4) #0.3=>boundary condition, :,1 means second collumn is chosen
print(y_pred_03)#printing the predicted result

#draw the ocnfusion matris from the outyput model
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y,y_pred)
print(cm)

#homework=> create confusion matric for y_pred where custom boundary conditions is being set

#performance matris
#calculate the presion and recall values from the ocnfusion matris

from sklearn.metrics import precision_score,recall_score
ps=precision_score(Y,y_pred)
rs=recall_score(Y,y_pred)

#show the AOC graph

from sklearn.metrics import roc_auc_score
rac=roc_auc_score(Y,y_pred)

print(ps,rs,rac,end="\n")

"""
#LDA model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_m=LinearDiscriminantAnalysis()
lda_m.fit(X,y)
y_pred_lda=lda_m.predict(X)
confusion_matrix(y,y_pred_lda)
"""

#test train curve for KNN
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
print("new shapes are:",X_train.shape,y_train.shape,X_test.shape,y_test.shape)
clf_LR=LogisticRegression()
clf_LR.fit(X_train,y_train)

#KNN model=> K Nerest Model
#note=> if a classifier then convert continous to discrete
#       if regressor then convert disctere to continous
#standardize the independent variable=> convert mean=0 and standardvariation=1
from sklearn import preprocessing
scaler=preprocessing.StandardScaler().fit(X_train)
X_train_s=scaler.transform(X_train)
scaler=preprocessing.StandardScaler().fit(X_test)
X_test_s=scaler.transform(X_test)

#train uour model to KNN
from sklearn.neighbors import KNeighborsClassifier
clf_knn1=KNeighborsClassifier(n_neighbors=1) #n_neighbours denotes the value of K
#fit the values in ypur KNN model
clf_knn1.fit(X_train_s,y_train)
#create a confusion matris for it
confusion_matrix(y_test,clf_knn1.predict(X_test_s))

#calculate the AUC score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,clf_knn1.predict(X_test_s))

#homework=> train your model for K=3

#creating single model for multiple vgalues of K
from sklearn.model_selection import GridSearchCV
#create a dictionary of parametrs
param={'n_neighbors':[1,3,4,5]}

#create and object and create it using our varaibles
grid_search_cv=GridSearchCV(KNeighborsClassifier(),param)
grid_search_cv.fit(X_train_s,y_train)

#find out the best suitable value of K among the param, and find its confusion matris and accuracy
print("best parameter is: ",grid_search_cv.best_params_)

best_model=grid_search_cv.best_estimator_
Y_pred_best=best_model.predict(X_test_s)
print(confusion_matrix(y_test,Y_pred_best))
print(accuracy_score(y_test,Y_pred_best))

#results of the file dataimport
""" 1)p value tells about the confidence level that is  do variables are dependent too much if the p value is too small  
    2) for estimate
        -ve sign indicates that the realtion is inverse proportion
        +ve sign indicates that there is direct proportion relation
    3) drawbacks of KNN
        doesn't tell about the individual dependencies of variables

    4) whenever there is a linear boundary that classifies the dataset=>logistic and LDA both perform well
        whenever there is a non linear boundary=> KNN performs better
"""

#summary of the  models used in dataimport
"""
results.

The first step is to do data collection.

You have to identify all the relevant variables and collect data for the same.

Once you've collected all the relevant data.

You have to pre-process it.

You learned how to do data preprocessing.

Few of the major steps that we took our outlire treatment in which we found out the outlying values

and change their values so that they do not harmfully impact our analysis.

Then we have missing value in imputation where we replaced blank values with harmless values such

as mean or medians.

We also did variable transformation.

We combined four different distance variables into one variable and so on.

So data preprocessing is a very important part.

You have to clean your data.

You have to put it into a tabular format with all the values of the variables in proper format so that

your model can work on it.

Next is model training.

If you have only one data set.

You have to split it into test and train data set.

You will use the training data set to train the model and we will use the test set to test its performance.

We have created the templates for logistic regression, linear discriminant analysis and KNN.

save those templates.

Whenever you face any business problem, just replace the dataset and you are good to go.

You can train your model with those same templates.

third point I've written here is do iterations.

The point is, when we trained our model before that we have taken some decisions on our data.

For example, we decided that we will impute the missing values using mean.

What will be the impact of using median?

Will that perform better?

We decided in variable transformation that we will replace these four distances by average distance.

Well, maybe it would make more business sense if replace it by the largest distance or the smallest

distance.

So we should do iterations of all these changes wherever we make our decision.

Lastly, when we are training the model, we should also compare the performance of different methods.

For example, here we learned three methods.

So we should compare the performance of all these three methods using a test set.

We saw in the last video how to do that

We use the confusion metrics for classification problems.

Draw the classification metrics of data set for all the different models that you have created and select

the best one.

So that's the last point.

We have to select the best model.

As I told you, there are two types of business problems.

One is prediction problem.

where our aim is to have maximum prediction accuracy. In such a case

We should use the model with best accuracy.

And the second type of problem is interpretation problem.

That is, we want to identify the relationship between a particular prediction variable

And the response variable.

For that, we can use the coefficient values of the parametric models.

Once we have selected the best model, for example, say linear discriminant analysis is giving us the

best prediction results.

And we have selected that model.

Now, whenever we get new data or new observations, we can feed those observations as a test set to

our model and find out the predicted classes for those observations.

So this is the whole process for a given data.

We train the model to start predicting.

We identify the model which is giving us the best predictions.

And once we have that model, which is giving us the best predictions, we use it to predict for future
"""

