library(dplyr)
library(ggplot2)
library(dplyr)
library(readr)

library(caret)
library(class)
library(DMwR)
library(dplyr)
library(e1071)
library(FSelectorRcpp)
library(imbalance)
library(rpart)
library(rpart.plot)

#importing data

train <- read_csv("Train-1542865627584.csv")
train$fraud <- ifelse(train$PotentialFraud=='Yes', 1, 0)
View(train)

bene <- read_csv("Train_Beneficiarydata-1542865627584.csv")
bene$RenalDiseaseIndicator<- ifelse(bene$RenalDiseaseIndicator=='Y', 1, 0)
for (i in 11:21){
  bene[, i]<- ifelse(bene[, i]==2, 0, 1)
}
View(bene)

inpa <- read_csv("Train_Inpatientdata-1542865627584.csv")
inpa$in_hospital <- as.numeric(as.Date(inpa$DischargeDt)- as.Date(inpa$AdmissionDt))
View(inpa)

outpa <- read_csv("Train_Outpatientdata-1542865627584.csv")
View(outpa)

#joining in and out with fraud status

inpa_fn <- inpa %>% inner_join(train, by = 'Provider')
#inpa_fn <- inpa_fn %>% rowwise %>% mutate(na_diagonosis = sum(is.na(inpa_fn[15:24])))
inpa_fn$no_dia_count <- apply(inpa_fn[15:24], 1, function(x) 10 -sum(is.na(x)))
inpa_fn$no_pro_count <- apply(inpa_fn[25:30], 1, function(x) 6-sum(is.na(x)))
inpa_fn$no_phy_count <- apply(inpa_fn[7:9], 1, function(x) 3-sum(is.na(x)))

View(inpa_fn)

# inpa_fn <- 



outpa_fn <- outpa %>% inner_join(train, by = 'Provider')
outpa_fn$no_dia_count <- apply(outpa_fn[10:19], 1, function(x) 10 -sum(is.na(x)))
outpa_fn$no_pro_count <- apply(outpa_fn[20:25], 1, function(x) 6-sum(is.na(x)))
outpa_fn$no_phy_count <- apply(outpa_fn[7:9], 1, function(x) 3-sum(is.na(x)))
outpa_fn$cl_di <- apply(outpa_fn[27], 1, function(x) 1-sum(is.na(x)))
View(outpa_fn)


# checking correlation
cor(inpa_fn$fraud,inpa_fn[34:36])
cor(inpa_fn$fraud,inpa_fn[31])

# 
glimpse(inpa_fn)
summary(inpa_fn %>% filter(fraud == 1))

#model
#knn
library(caret)

#
normalize = function(x){
  return ((x - min(x))/(max(x) - min(x)))}
# normalize
Inpa_normalized = inpa_fn %>% mutate_at(c(6,31,33,34,35,36), normalize)
inpa_2 = Inpa_normalized[c(6,31,33,34,35,36)]
library(tidyr)
inpa_2<-inpa_2 %>% drop_na()
unique(inpa_2$fraud)
inpa_2$fraud = as.factor(inpa_2$fraud)
# Randomly pick the rows for training partition
train_rows = createDataPartition(y = inpa_2$fraud, p = 0.70, list = FALSE)
inpa_train = inpa_2[train_rows,]
inpa_test = inpa_2[-train_rows,]
# "-" means selecting rows *not* included in train_rows
inpa_train<-inpa_train %>% drop_na()
inpa_test<-inpa_test %>% drop_na()
# Oversampling
new_train = oversample(inpa_train, ratio = 0.5, method = "ADASYN", classAttr = 3)
# Train a Decision Tree Model
tree = rpart(fraud ~ ., data = inpa_train,
             method = "class",
             parms = list(split = "information"),
             control = list(minsplit = 1,
                            maxdepth = 3,
                            cp = 0.00004))   # useful -1,3,0.00004 -0.7189; 2,4,0.00004 - 0.7186 ;  1,2,0.00004 -0.7184   ; 2,1,0.00004 -0.7161
# Evaluate Decision Tree Performance
pred_tree = predict(tree, inpa_test, type = "class")
confusionMatrix(data = pred_tree,
                reference = inpa_test$fraud,
                mode = "prec_recall",
                positive = "1")
# Visualize Decision Tree
prp(tree, varlen = 0)


inpa_3 = inpa_fn[c(6,31,33,34,35,36)]
library(tidyr)
inpa_3<-inpa_3 %>% drop_na()
unique(inpa_3$fraud)
inpa_3$fraud = as.factor(inpa_3$fraud)
# Randomly pick the rows for training partition
train_rows = createDataPartition(y = inpa_3$fraud, p = 0.70, list = FALSE)
inpa_train_3 = inpa_3[train_rows,]
inpa_test_3 = inpa_3[-train_rows,]
# "-" means selecting rows *not* included in train_rows
inpa_train_3<-inpa_train_3 %>% drop_na()
inpa_test_3<-inpa_test_3 %>% drop_na()
# Oversampling
new_train = oversample(inpa_train_3, ratio = 0.5, method = "ADASYN", classAttr = 3)
# Train a Decision Tree Model
tree = rpart(fraud ~ ., data = inpa_train_3,
             method = "class",
             parms = list(split = "information"),
             control = list(minsplit = 1,
                            maxdepth = 3,
                            cp = 0.00004))   
#  1, 3, 0.00004    0.7272
# useful -1,3,0.00004 -0.7189; 2,4,0.00004 - 0.7186 ;  1,2,0.00004 -0.7184   ; 2,1,0.00004 -0.7161
# Evaluate Decision Tree Performance
pred_tree = predict(tree, inpa_test_3, type = "class")
confusionMatrix(data = pred_tree,
                reference = inpa_test_3$fraud,
                mode = "prec_recall",
                positive = "1")
# Visualize Decision Tree
prp(tree, varlen = 0)


# for out data

outpa_3 = outpa_fn[c(6,26, 29,30,31,32,33)]
library(tidyr)
outpa_3<-outpa_3 %>% drop_na()
unique(outpa_3$fraud)
outpa_3$fraud = as.factor(outpa_3$fraud)
# Randomly pick the rows for training partition
train_rows = createDataPartition(y = outpa_3$fraud, p = 0.70, list = FALSE)
outpa_train_3 = outpa_3[train_rows,]
outpa_test_3 = outpa_3[-train_rows,]
# "-" means selecting rows *not* included in train_rows
outpa_train_3<-outpa_train_3 %>% drop_na()
outpa_test_3<-outpa_test_3 %>% drop_na()
# Oversampling
new_train = oversample(outpa_train_3, ratio = 0.5, method = "ADASYN", classAttr = 3)
# Train a Decision Tree Model
tree = rpart(fraud ~ ., data = outpa_train_3,
             method = "class",
             parms = list(split = "information"),
             control = list(minsplit = 1,
                            maxdepth = 4,
                            cp = 0.000000004))   
# useful-  ,1,5,0.000004 - 2.111e-04;1,4,0.000004 - 2.815e-04 ; 1,3,0.000004 - 3.520e-05  ;
#1, 3, 0.00000004 - 3.520e-05;  (c(6,26, 29,30,31,32,33)); 
# 2, 3, 0.000004 - 2.112e-04;  (c(6,26, 29,30,31,32))
# Evaluate Decision Tree Performance
pred_tree = predict(tree, outpa_test_3, type = "class")
confusionMatrix(data = pred_tree,
                reference = outpa_test_3$fraud,
                mode = "prec_recall",
                positive = "1")
# Visualize Decision Tree
prp(tree, varlen = 0)

#

inpa_ana <- inpa_3 %>% filter( InscClaimAmtReimbursed > 14000)
inpa_ana %>% group_by(fraud) %>% tally()
inpa_ana %>% add_count(fraud, wt = runs)





#


library(caret)
library(class)
library(DMwR)
library(dplyr)
library(e1071)
library(FSelectorRcpp)
library(imbalance)
library(rpart)
library(rpart.plot)

XYZData = read.csv("XYZData.csv")
XYZData$adopter = as.factor(XYZData$adopter)
XYZData = XYZData %>% select(-user_id)
# Training/Validation Split
train_rows = createDataPartition(y = XYZData$adopter, p = 0.70, list = FALSE)
XYZData_train = XYZData[train_rows,]
XYZData_test = XYZData[-train_rows,]
# Oversampling
newXYZData_train = oversample(XYZData_train, ratio = 0.5, method = "ADASYN", classAttr = "adopter")
# Train a Decision Tree Model
tree = rpart(adopter ~ ., data = newXYZData_train,
             method = "class",
             parms = list(split = "information"),
             control = list(minsplit = 5,
                            maxdepth = 10,
                            cp = 0.004))
# Evaluate Decision Tree Performance
pred_tree = predict(tree, XYZData_test, type = "class")
confusionMatrix(data = pred_tree,
                reference = XYZData_test$adopter,
                mode = "prec_recall",
                positive = "1")
# Visualize Decision Tree
prp(tree, varlen = 0)



## KNN

XYZData = read.csv("XYZData.csv")
XYZData$adopter = as.factor(XYZData$adopter)
XYZData = XYZData %>% select(-user_id)
# Training/Validation Split
train_rows = createDataPartition(y = XYZData$adopter, p = 0.70, list = FALSE)
XYZData_train = XYZData[train_rows,]
XYZData_test = XYZData[-train_rows,]
# Oversampling
newXYZData_train = oversample(XYZData_train, ratio = 0.5, method = 'ADASYN', classAttr = 'adopter')
# Normalize
normalize = function(x){
  return ((x - min(x))/(max(x) - min(x)))}
XYZData_normalized_train = newXYZData_train %>% mutate_at(1:10, normalize)
XYZData_normalized_test = XYZData_test %>% mutate_at(1:10, normalize)
# Train a k-NN Model
pred_knn = knn(train = XYZData_normalized_train[,1:10],
               test = XYZData_normalized_test[,1:10],
               cl = XYZData_normalized_train$adopter,
               k = 26)
# Evaluate k-NN Model Performance
confusionMatrix(data = pred_knn,
                reference = XYZData_normalized_test$adopter,
                mode = "prec_recall",
                positive = "1")

## Feature Selection

for (kval in 5:20){
  XYZData = read.csv("XYZData.csv")
  XYZData$adopter = as.factor(XYZData$adopter)
  XYZData = XYZData %>% select(-user_id)
  
  train_rows = createDataPartition(y = XYZData$adopter, p = 0.70, list = FALSE)
  XYZData_train = XYZData[train_rows,]
  XYZData_test = XYZData[-train_rows,]
  IG = information_gain(adopter ~ ., data = XYZData_train)
  topK = cut_attrs(IG, k = kval)
  XYZData_train = XYZData_train %>% select(topK, adopter)
  XYZData_test = XYZData_test %>% select(topK, adopter)
  
  newXYZData_train = oversample(XYZData_train, ratio = 0.5, method = "ADASYN", classAttr = "adopter")
  
  XYZData_normalized_train = newXYZData_train %>% mutate_at(1:kval, normalize)
  XYZData_normalized_test = XYZData_test %>% mutate_at(1:kval, normalize)
  
  pred_knn = knn(train = XYZData_normalized_train[,1:kval],
                 test = XYZData_normalized_test[,1:kval],
                 cl =  XYZData_normalized_train$adopter,
                 k = 26)
  
  cm = confusionMatrix(data = pred_knn, XYZData_normalized_test$adopter, mode = "prec_recall", positive = "1")
  F1 = cm$byClass["F1"]
  print(F1)
}

## Try a few other k values

for (kval in 2:30){
  pred_knn = knn(train = XYZData_normalized_train[,1:10],
                 test = XYZData_normalized_test[,1:10],
                 cl =  XYZData_normalized_train$adopter,
                 k = kval)
  cm = confusionMatrix(data = pred_knn, XYZData_normalized_test$adopter, mode = "prec_recall", positive = "1")
  F1 = cm$byClass["F1"]
  print(F1)
}
## Naive Bayes

XYZData = read.csv("XYZData.csv")
XYZData$adopter = as.factor(XYZData$adopter)
XYZData = XYZData %>% select(-user_id)
# Training/Validation Split
train_rows = createDataPartition(y = XYZData$adopter, p = 0.70, list = FALSE)
XYZData_train = XYZData[train_rows,]
XYZData_test = XYZData[-train_rows,]
# Oversampling
newXYZData_train = oversample(XYZData_train, ratio = 0.5, method = 'ADASYN', classAttr = 'adopter')
table(newXYZData_train$adopter)
# Train a Naive Bayes Model
NB_model = naiveBayes(adopter ~ ., data = newXYZData_train)
# Make Predictions
pred_nb = predict(NB_model, XYZData_test)
prob_pred_nb = predict(NB_model, XYZData_test, type = "raw")
# Evaluate Model Performance
confusionMatrix(data = pred_nb,
                reference = XYZData_test$adopter,
                mode = "prec_recall",
                positive = "1")

