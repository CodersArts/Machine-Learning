library(plyr)
library(corrplot)
library(ggplot2)
library(gridExtra)
library(ggthemes)
library(caret)
library(MASS)
library(randomForest)
library(party)

df_train <- read.csv('Train_data.csv')
df_test <- read.csv('Test_data.csv')

#Remove the columns we do not need for the analysis in training data
df_train$state <- NULL
df_train$area.code <- NULL
df_train$phone.number <- NULL


#Remove the columns we do not need for the analysis in testing data
df_test$state <- NULL
df_test$area.code <- NULL
df_test$phone.number <- NULL

## Converting the target variable to categorical variable

df_train$Churn=factor(df_train$Churn)
df_test$Churn=factor(df_test$Churn)

##Fitting the Model


#Random Forest model
rfModel <- randomForest(Churn ~., data = df_train)
print(rfModel)

pred_rf <- predict(rfModel, df_test)
print(pred_rf)


