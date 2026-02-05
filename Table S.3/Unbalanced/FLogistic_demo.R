# flrg
library(fda)
library(refund)

train_class1 = 50
train_class2 = 250
n = 1024
test_class1 = 25
test_class2 = 25
total_num_test = test_class1 + test_class2
total_num_train = train_class1 + train_class2

Y = as.factor(c(rep(0, train_class1), rep(1, train_class2)))
Y_test = as.factor(c(rep(0, test_class1), rep(1, test_class2)))
case_res = list()
basis_num = 5


for(case_num in 1:3){
  sum_res = c()
  print(paste("Model", case_num))
  for(simu_num in 1:500){ 
    X_li_eval = list()
    for(i in 1:total_num_train){
      if(i <= train_class1){
        X_li_eval[[i]] = get(function_names[2*case_num - 1])(n, 1)
      } else{
        X_li_eval[[i]] = get(function_names[2*case_num])(n, 1)
      }
    }
    df_train <- do.call(rbind, lapply(X_li_eval, function(x) data.frame(t(x))))
    df_train = t(as.matrix(df_train))
    
    X_li_test = list()
    for(i in 1:total_num_test){
      if(i <= 25){
        X_li_test[[i]] = get(function_names[2*case_num - 1])(n, 1)
      } else{
        X_li_test[[i]] = get(function_names[2*case_num])(n, 1)
      }
    }
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
    sum_res[simu_num] = sum(as.vector(yhat) == Y_test)/50
  }
  
  case_res[[case_num]] = sum_res
  print(c(mean(sum_res), sd(sum_res)))
  
}

# Can also output together. The output is inorder for Model 1 to Model 6
# unlist(lapply(case_res, mean))
# unlist(lapply(case_res, sd))
