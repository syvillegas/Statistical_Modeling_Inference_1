

# We add this command to remove the environment every time,
# so that we don't get unexpected problems.
rm(list = ls())

# We set the source code to be able to use the given functions
# in the seminar folder shared in CLassroom, with several modifications
# done in those functions so that they can be used for 
# 'generalized linear models (GLMs).
source('routines_seminar1_mod.R')

# We load the data on which we are going to be working.
spam <- read.table(file.path('data/spam.data'), quote="\"", comment.char="")
spam.names <- c(read.table("data/spambase.names", sep = ":", skip = 33, 
                          nrows = 53, as.is = TRUE) [,1], "char_freq_#", 
                          read.table("data/spambase.names", sep = ":", 
                          skip = 87, nrows = 3,as.is = TRUE)[, 1], "spam.01")
names(spam) <- spam.names
spam <- spam[sample(nrow(spam)),]

# We set a random seed
seed=2345

# We get the last column of the DataFrame, in which we have
# 1s and 0s, in order to identify the e-mails that are, and are not, 
# spam e-mails, respectively. We define this extracted column as y_train.
# We then create X_train by removing y_train from the original DataFrame.

y_train <- spam['spam.01']
X_train <- subset(spam, select=-spam.01)

# We also convert both y and the DataFrame into matrices, so that
# we can operate with them in the given functions.
y_train_m <- as.matrix(y_train)
X_train_m <- as.matrix(X_train)

#LASSO BIC
# We begin by fitting the model, in order to obtain 
# the number of variables we will be considering. 
# We use the modified function,in which the family of 
# the distribution is a parameter of the function.

fit.lassobic = lasso.bic_mod(x=X_train_m, y=y_train_m, family = "binomial")

# In order to determine the number or variables, or predictors,
# that we will have, we count the number of coefficients with a
# non-zero value.
coef_lassobic <- fit.lassobic$coef
names(coef_lassobic) <- c('intercept', colnames(X_train))
lassobic_vars = length(coef_lassobic[coef_lassobic!=0])
# From the obtained value, we have to substract 1, due to 
# the intercept value which is also being counted.
########

# We get the cross-validated predictions for a LASSO regression with the given
# function, which has been modified in order to be able to pass the family
# of the distribution as a parameter of the function. 
lasso_bic = kfoldCV.lasso_mod(x=X_train_m, y=y_train_m,K=10,seed=seed,
                          criterion="bic", family = "binomial", 
                          predict_type = "response")

# We define a variable with the recently obtained predictions
# and we convert them from probability predictions to binary ones,
# by creating a new variable that, if the probability is higher than 0.5,
# 1 is appended to the new binary vector of predictions. Otherwise,
# 0 is appended.
pred_prob_bic <- lasso_bic$pred
pred_bin_bic <- c()
for (i in pred_prob_bic){
  if (i>0.5){
    pred_bin_bic <- c(pred_bin_bic, 1)
  }
  else{
    pred_bin_bic <- c(pred_bin_bic,0)
  }
}
# Finally, we compute the R^2 of the model with the following expression
rsquared_lasso_bic = 1-(sum((y_train_m-pred_bin_bic)^2)/sum((y_train_m-mean(y_train_m))^2))
################################

#LASSO CV
# We begin by fitting the model, in order to obtain 
# the number of variables we will be considering. 
# We use the cv.glmnet function via the 10-fold cross-validation.
fit.lassocv = cv.glmnet(x=X_train_m, y=y_train_m, nfolds=10, 
                        family='binomial')

# In order to determine the number or variables, or predictors,
# that we will have, we count the number of coefficients with a
# non-zero value.
coef_lassocv <- as.vector(coef(fit.lassocv, s='lambda.min'))
names(coef_lassocv) <- c('intercept', colnames(X_train))
lassocv_vars = length(coef_lassocv[coef_lassocv!=0])
# From the obtained value, we have to substract 1, due to 
# the intercept value which is also being counted.

# We get the cross-validated predictions for a LASSO regression with the given
# function, which has been modified in order to be able to pass the family
# of the distribution as a parameter of the function. 

lasso_cv = kfoldCV.lasso_mod(x=X_train_m, y=y_train_m,K=10,seed=seed,
                          criterion="cv", family = "binomial", 
                          predict_type = "response")
# We convert the obtained probability predictions to binary ones,
# the same way we did before.
pred_prob_cv <- lasso_cv$pred
pred_bin_cv <- c()
for (i in pred_prob_cv){
  if (i>0.5){
    pred_bin_cv <- c(pred_bin_cv, 1)
  }
  else{
    pred_bin_cv <- c(pred_bin_cv,0)
  }
}
# Finally, we compute the R^2 of the model
rsquared_lasso_cv = 1-(sum((y_train_m-pred_bin_cv)^2)/sum((y_train_m-mean(y_train_m))^2))
rsquared_lasso_cv
################################

# MLE
# MLE always takes into account all of the predictors, therefore we directly
# focus in getting the predictions.
# We get the cross-validated predictions for the least-squares regression
# with the given function, which has been modified in order to be able to pass
# the family of the distribution as a parameter of the function. 

mle = kfoldCV.mle_mod(x=X_train, y=y_train_m,K=10,seed=seed,
                         family = "binomial", 
                         predict_type = "response")
# We convert the obtained probability predictions to binary ones,
# the same way we did before.
pred_prob_mle <- mle$pred
pred_bin_mle <- c()
for (i in pred_prob_mle){
  if (i>0.5){
    pred_bin_mle <- c(pred_bin_mle, 1)
  }
  else{
    pred_bin_mle <- c(pred_bin_mle,0)
  }
}
# Finally, we compute the R^2 of the model
rsquared_mle = 1-(sum((y_train_m-pred_bin_mle)^2)/sum((y_train_m-mean(y_train_m))^2))
rsquared_mle
################################









