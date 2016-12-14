## take a look some of the data available

prop.table(table(traindata$Sex))
#    female      male 
# 0.3618677 0.6381323 

prop.table(table(traindata$Survived))
#         0         1 
# 0.5894942 0.4105058 

prop.table(table(traindata$Sex, traindata$Survived))      
#                 0          1
# female 0.08949416 0.27237354
# male   0.50000000 0.13813230

## install glmnet to make use of LASSO, Ridge, and Elasticnet

install.packages("glmnet", repos = "http://cran.us.r-project.org")
require(glmnet)

## Run LASSO

# Sex field needs to be numeric to run binomial family for LASSO
traindata.factored <- traindata
traindata.factored$Sex <- as.numeric(traindata.factored$Sex)
testdata.factored <- testdata
testdata.factored$Sex <- as.numeric(testdata.factored$Sex)

# get the LASSO model
lasso.model <- glmnet(as.matrix(traindata.factored[,2:5]), as.matrix(traindata.factored[,1]), family="binomial", alpha=1)

# get cross-validated LASSO model to estimate the optimal lambda
cv.lasso.model <- cv.glmnet(as.matrix(traindata.factored[,2:5]), as.matrix(traindata.factored[,1]), family="binomial", alpha=1)

# estimated optimal lambda
cv.lasso.model$lambda.min # 0.00245966

plot(lasso.model, label=TRUE)
plot(cv.lasso.model)

# get the ROC curve of the cross-validated LASSO model
auc.cv.lasso.model <- cv.glmnet(x = as.matrix(traindata.factored[,2:5]), y = as.matrix(traindata.factored[,1]), family="binomial", alpha=1, type.measure="auc")
test.prob <- predict(auc.cv.lasso.model, type="response", newx = as.matrix(testdata.factored[,2:5]), s = 'lambda.min')

# predict the test data on whether someone survived or not
library(ROCR)
require(ROCR)

pred.auc <- prediction(test.prob[,1], testdata.factored[,1])

# obtain the false positive and true positive rate of the prediction
cv.lasso.auc.fp <- slot(pred.auc, "fp")
cv.lasso.auc.tp <- slot(pred.auc, "tp")
cv.lasso.fpr <- unlist(cv.lasso.auc.fp) / unlist(slot(pred.auc, "n.neg"))
cv.lasso.tpr <- unlist(cv.lasso.auc.tp) / unlist(slot(pred.auc, "n.pos"))
cv.lasso.perf.auc = performance(pred.auc, "auc")
cv.lasso.auc <- cv.lasso.perf.auc@y.values[[1]]

# plot the ROC curve
plot(cv.lasso.fpr, cv.lasso.tpr, main="ROC Curve from cross-validated LASSO regression", xlab="False Positive Rate", ylab="True Positive Rate")
# label the AUC on the plot for easy viewing
text(0.4, 0.6, paste("LASSO AUC = ", format(cv.lasso.auc, digits=5, scientific=FALSE)))


## Run Ridge

# get the Ridge model
# NOTE: alpha=0 (alpha=1 in LASSO)
ridge.model <- glmnet(as.matrix(traindata.factored[,2:5]), as.matrix(traindata.factored[,1]), family="binomial", alpha=0)

# get cross-validated Ridge model to estimate the optimal lambda
cv.ridge.model <- cv.glmnet(as.matrix(traindata.factored[,2:5]), as.matrix(traindata.factored[,1]), family="binomial", alpha=0)

# estimated optimal lambda
cv.ridge.model$lambda.min # 0.02828012

plot(ridge.model, label=TRUE)
plot(cv.ridge.model)

# get the ROC curve of the cross-validated Ridge model
auc.cv.ridge.model <- cv.glmnet(x = as.matrix(traindata.factored[,2:5]), y = as.matrix(traindata.factored[,1]), family="binomial", alpha=1, type.measure="auc")
ridge.test.prob <- predict(auc.cv.ridge.model, type="response", newx = as.matrix(testdata.factored[,2:5]), s = 'lambda.min')

# predict the test data on whether someone survived or not
ridge.pred.auc <- prediction(ridge.test.prob[,1], testdata.factored[,1])

# obtain the false positive and true positive rate of the prediction
cv.ridge.auc.fp <- slot(ridge.pred.auc, "fp")
cv.ridge.auc.tp <- slot(ridge.pred.auc, "tp")
cv.ridge.fpr <- unlist(cv.ridge.auc.fp) / unlist(slot(ridge.pred.auc, "n.neg"))
cv.ridge.tpr <- unlist(cv.ridge.auc.tp) / unlist(slot(ridge.pred.auc, "n.pos"))
cv.ridge.perf.auc = performance(ridge.pred.auc, "auc")
cv.ridge.auc <- cv.ridge.perf.auc@y.values[[1]]

# plot the ROC curve
plot(cv.ridge.fpr, cv.ridge.tpr, main="ROC Curve from cross-validated Ridge regression", xlab="False Positive Rate", ylab="True Positive Rate")
# label the AUC on the plot for easy viewing
text(0.4, 0.6, paste("Ridge AUC = ", format(cv.ridge.auc, digits=5, scientific=FALSE)))


## Run Elasticnet

# get the Elasticnet model
# NOTE: alpha=0.5 (alpha=1 in LASSO, alpha=0 in Ridge)
elasticnet.model <- glmnet(as.matrix(traindata.factored[,2:5]), as.matrix(traindata.factored[,1]), family="binomial", alpha=0.5)

# get cross-validated Elastic Net model to estimate the optimal lambda
cv.elasticnet.model <- cv.glmnet(as.matrix(traindata.factored[,2:5]), as.matrix(traindata.factored[,1]), family="binomial", alpha=0.5)

# estimated optimal lambda
cv.elasticnet.model$lambda.min # 0.004084106

plot(elasticnet.model, label=TRUE)
plot(cv.elasticnet.model)

# get the ROC curve of the cross-validated Elastic Net model
auc.cv.elasticnet.model <- cv.glmnet(x = as.matrix(traindata.factored[,2:5]), y = as.matrix(traindata.factored[,1]), family="binomial", alpha=1, type.measure="auc")
elasticnet.test.prob <- predict(auc.cv.elasticnet.model, type="response", newx = as.matrix(testdata.factored[,2:5]), s = 'lambda.min')

# predict the test data on whether someone survived or not
elasticnet.pred.auc <- prediction(elasticnet.test.prob[,1], testdata.factored[,1])

# obtain the false positive and true positive rate of the prediction
cv.elasticnet.auc.fp <- slot(elasticnet.pred.auc, "fp")
cv.elasticnet.auc.tp <- slot(elasticnet.pred.auc, "tp")
cv.elasticnet.fpr <- unlist(cv.elasticnet.auc.fp) / unlist(slot(elasticnet.pred.auc, "n.neg"))
cv.elasticnet.tpr <- unlist(cv.elasticnet.auc.tp) / unlist(slot(elasticnet.pred.auc, "n.pos"))
cv.elasticnet.perf.auc = performance(elasticnet.pred.auc, "auc")
cv.elasticnet.auc <- cv.elasticnet.perf.auc@y.values[[1]]

# plot the ROC curve
plot(cv.elasticnet.fpr, cv.elasticnet.tpr, main="ROC Curve from cross-validated Elastic Net regression", xlab="False Positive Rate", ylab="True Positive Rate")
# label the AUC on the plot for easy viewing
text(0.4, 0.6, paste("Elastic Net AUC = ", format(cv.elasticnet.auc, digits=5, scientific=FALSE)))
