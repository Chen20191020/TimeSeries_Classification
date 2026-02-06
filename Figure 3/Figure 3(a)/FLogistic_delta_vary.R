# flrg
library(fda)
library(refund)

generate_AR1_test_class1 = function(n, v, delta){
  ts = c()
  x_ini = rnorm(1,0,1)
  for(i in 1:n){
    if(i == 1){
      ts[i] = 2*delta*cos(2*pi*(i/n))*x_ini  + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] = 2*delta*cos(2*pi*(i/n))*ts[i-1] + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}

generate_AR1_test_class2 = function(n, v, delta){
  ts = c()
  x_ini = rnorm(1,0,1)
  for(i in 1:n){
    if(i == 1){
      ts[i] = delta*cos(2*pi*(i/n))*x_ini + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] = delta*cos(2*pi*(i/n))*ts[i-1] + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}


train_class1 = 100
train_class2 = 100
n = 1024
test_class1 = 25
test_class2 = 25
total_num_test = test_class1 + test_class2
total_num_train = train_class1 + train_class2

Y = as.factor(c(rep(0, train_class1), rep(1, train_class2)))
Y_test = as.factor(c(rep(0, test_class1), rep(1, test_class2)))
case_res = list()
basis_num = 5

n_del= seq(0.05, 0.4, by = 0.05)

for(ind_del in 1:length(n_del)){
  sum_res = c()
  for(simu_num in 1:500){ 
    X_li_eval = list()
    for(i in 1:total_num_train){
      if(i <= train_class1){
        X_li_eval[[i]] = generate_AR1_test_class1(n, 1, n_del[ind_del])
      } else{
        X_li_eval[[i]] = generate_AR1_test_class2(n, 1, n_del[ind_del])
      }
    }
    df_train <- do.call(rbind, lapply(X_li_eval, function(x) data.frame(t(x))))
    df_train = t(as.matrix(df_train))
    
    X_li_test = list()
    for(i in 1:total_num_test){
      if(i <= 25){
        X_li_test[[i]] = generate_AR1_test_class1(n, 1, n_del[ind_del])
      } else{
        X_li_test[[i]] = generate_AR1_test_class2(n, 1, n_del[ind_del])
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
    lfreg.model = lfreg(formula, betalist = betalist)
    
    # Prediction on new data
    newdata = list(df_test)
    # newdata = list(xfd_1, latitude, longitude)
    yhat = predict(lfreg.model, newdata = newdata, type = "labels")
    sum_res[simu_num] = sum(as.vector(yhat) == Y_test)/50
  }
  case_res[[ind_del]] = sum_res
  
  
}


# Print the accuracy for delta values from 0.05 to 0.40, in order.
round(unlist(lapply(case_res, mean)),3)

round(unlist(lapply(case_res, sd)),3)

