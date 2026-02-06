
# flrg
library(fda)
library(refund)

train_class1 = 252
train_class2 = 252
n = 4500
test_class1 = 75
test_class2 = 75
total_num_test = test_class1 + test_class2
total_num_train = train_class1 + train_class2

Y = as.factor(c(rep(0, train_class1), rep(1, train_class2)))
Y_test = as.factor(c(rep(0, test_class1), rep(1, test_class2)))
case_res = list()
basis_num = 5


u_dsample <- function(data, target_size = 4500) {
  n <- length(data)
  if (n <= target_size) {
    stop("Input data must have more than 4096 points.")
  }
  
  indices <- round(seq(1, n, length.out = target_size))
  return(data[indices])
}

#  classfication for EEG 

# C3 train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_C3.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_C3.rds")


for(j in 1:504){
  X_li_train[[j]] = u_dsample(X_li_train[[j]])
}

for(j in 1:150){
  X_li_test[[j]] = u_dsample(X_li_test[[j]])
}


df_train <- do.call(rbind, lapply(X_li_train, function(x) data.frame(t(x))))
df_train = t(as.matrix(df_train))

df_test <- do.call(rbind, lapply(X_li_test, function(x) data.frame(t(x))))
df_test = t(as.matrix(df_test))

x = df_train
xbasis = create.bspline.basis(c(1,n), basis_num) # 5 basis functions
xfd = smooth.basis(c(1:n),x, xbasis)$fd

bbasis = create.bspline.basis(c(0,n), basis_num)
betalist = list(bbasis)
formula = Y ~ xfd
options(warn = -1)
lfreg.model = lfreg(formula, betalist = betalist)

# Prediction on new data
newdata = list(df_test)
# newdata = list(xfd_1, latitude, longitude)
yhat = predict(lfreg.model, newdata = newdata, type = "labels")
y_hat_c3 = yhat 
print(paste("Channel C3 accuracy:", round(round(sum(as.vector(yhat) == Y_test)/150, 3), 3)))


# C4 train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_C4.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_C4.rds")

# classification  

for(j in 1:504){
  X_li_train[[j]] = u_dsample(X_li_train[[j]])
}

for(j in 1:150){
  X_li_test[[j]] = u_dsample(X_li_test[[j]])
}


df_train <- do.call(rbind, lapply(X_li_train, function(x) data.frame(t(x))))
df_train = t(as.matrix(df_train))

df_test <- do.call(rbind, lapply(X_li_test, function(x) data.frame(t(x))))
df_test = t(as.matrix(df_test))

x = df_train
xbasis = create.bspline.basis(c(1,n), basis_num) # 5 basis functions
xfd = smooth.basis(c(1:n),x, xbasis)$fd

bbasis = create.bspline.basis(c(0,n), basis_num)
betalist = list(bbasis)
formula = Y ~ xfd
options(warn = -1)
lfreg.model = lfreg(formula, betalist = betalist)

# Prediction on new data
newdata = list(df_test)
# newdata = list(xfd_1, latitude, longitude)
yhat = predict(lfreg.model, newdata = newdata, type = "labels")
y_hat_c4 = yhat
print(paste("Channel C4 accuracy:", round(round(sum(as.vector(yhat) == Y_test)/150, 3), 3)))



# CZ train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_CZ.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_CZ.rds")

# classification  
for(j in 1:504){
  X_li_train[[j]] = u_dsample(X_li_train[[j]])
}

for(j in 1:150){
  X_li_test[[j]] = u_dsample(X_li_test[[j]])
}


df_train <- do.call(rbind, lapply(X_li_train, function(x) data.frame(t(x))))
df_train = t(as.matrix(df_train))

df_test <- do.call(rbind, lapply(X_li_test, function(x) data.frame(t(x))))
df_test = t(as.matrix(df_test))

x = df_train
xbasis = create.bspline.basis(c(1,n), basis_num) # 5 basis functions
xfd = smooth.basis(c(1:n),x, xbasis)$fd

bbasis = create.bspline.basis(c(0,n), basis_num)
betalist = list(bbasis)
formula = Y ~ xfd
options(warn = -1)
lfreg.model = lfreg(formula, betalist = betalist)

# Prediction on new data
newdata = list(df_test)
# newdata = list(xfd_1, latitude, longitude)
yhat = predict(lfreg.model, newdata = newdata, type = "labels")
# classification  
y_hat_cz = yhat
print(paste("Channel CZ accuracy:", round(round(sum(as.vector(yhat) == Y_test)/150, 3), 3)))


# Majority voting 
# Combine the vectors into a matrix
matrix = cbind(as.vector(y_hat_cz), as.vector(y_hat_c3), as.vector(y_hat_c4))
# Compute the majority for each element
majority = rowSums(matrix) >= 2
# Convert logical TRUE/FALSE to numeric 1/0
majority = as.numeric(majority)
print(paste("Majority voting accuracy:", round(sum(majority == Y_test)/150, 3)))

