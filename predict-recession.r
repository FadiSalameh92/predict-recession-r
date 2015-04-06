
#  *------------------------------------------------------------------*
#  | feature selection
#  *------------------------------------------------------------------*
install.packages("caret", repos="http://R-Forge.R-project.org")
library(caret)

feature <- read.csv('D:/study/learning material/2015winter/data5000/project/data/feature.csv')
mdrrDescr <- feature[,3:12]
mdrrClass <- feature[,2]
Process <- preProcess(mdrrDescr)
str(mdrrDescr)
newdata3 <- predict(Process, mdrrDescr)

profile <- rfe(newdata3,mdrrClass,sizes = c(1,2,3,4,5,6,7,8,9,10),rfeControl = rfeControl(functions=rfFuncs,method='cv'))
plot(profile,type=c('o','g'))
print(profile)

#  *------------------------------------------------------------------*
#  | data partition
#  *------------------------------------------------------------------*
data <- read.csv('D:/study/learning material/2015winter/data5000/project/data/data.csv')
inTrain <- createDataPartition(y = data$recession,p = .75,list = FALSE)
training <- data[ inTrain,]
testing <- data[-inTrain,]
nrow(training)
nrow(testing)

#  *------------------------------------------------------------------*
#  | train with nnet
#  *------------------------------------------------------------------*
ctrl <- trainControl(method = "repeatedcv",repeats = 3,summaryFunction = twoClassSummary,classProbs = TRUE)
my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(5, 6, 7))
#myGrid <- data.frame(layer1 = 3, layer2=0,layer3=0)
#nn <- train(recession~spread+spindex+cbindex, data = training, method = "neuralnet",tuneGrid=myGrid,metric="ROC" trControl = ctrl,preProc = "range")
training$recession<-as.factor(training$recession)

set.seed(1)
nn <- train(recession~spread+spindex+cbindex, data = training, method = "nnet",tuneGrid=my.grid,metric="ROC" ,trControl = ctrl,preProc = "range")
nnresult <- predict(nn, newdata = testing)
nn
plot(nn)


#threshold <- 0.5
#pred <- factor( ifelse(nnresult > threshold, "1", "0") )
#pred <- relevel(pred, "1")
confusionMatrix(nnresult, testing$recession)


#  *------------------------------------------------------------------*
#  | train with svm
#  *------------------------------------------------------------------*
set.seed(2)
svmTune <- train(recession~spread+spindex+cbindex, data = training,method = "svmRadial",tuneLength = 9,preProc = c("center", "scale"),metric = "ROC",trControl = ctrl)
svmTune
plot(svmTune)
svmresult <- predict(svmTune, newdata = testing)
confusionMatrix(svmresult, testing$recession)

#  *------------------------------------------------------------------*
#  | train with knn
#  *------------------------------------------------------------------*
set.seed(3)
knnTune <- train(recession~spread+spindex+cbindex, data = training,method = "knn",tuneLength = 12,preProc = c("center", "scale"),metric = "ROC",trControl = ctrl)
knnTune
plot(knnTune)
knnresult <- predict(knnTune, newdata = testing)
plot(knnresult)
confusionMatrix(knnresult, testing$recession)

#  *------------------------------------------------------------------*
#  | resample models
#  *------------------------------------------------------------------*
cvValues <- resamples(list(NNET = nn, SVM = svmTune, KNN = knnTune))
summary(cvValues)
splom(cvValues, metric = "ROC")
xyplot(cvValues, metric = "ROC")
parallelplot(cvValues, metric = "ROC")
dotplot(cvValues, metric = "ROC")

#  *------------------------------------------------------------------*
#  | compare with models
#  *------------------------------------------------------------------*
rocDiffs <- diff(cvValues, metric = "ROC")
summary(rocDiffs)
dotplot(rocDiffs, metric = "ROC")
