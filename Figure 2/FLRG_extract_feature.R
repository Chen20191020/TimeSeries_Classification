library(fda)
library(refund)

generate_AR1_class1 = function(n, v){
  ts = c()
  x_ini = rnorm(1, 0, 1/v)
  for(i in 1:n){
    if(i == 1){
      ts[i] = 0.4*cos(2*pi*(i/n))*x_ini  + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] = 0.4*cos(2*pi*(i/n))*ts[i-1] + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}

generate_AR1_class2 = function(n, v){
  ts = c()
  x_ini = rnorm(1, 0, 1/v)
  for(i in 1:n){
    if(i == 1){
      ts[i] = 0.2*cos(2*pi*(i/n))*x_ini + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] = 0.2*cos(2*pi*(i/n))*ts[i-1] + rnorm(1, 0, 1/v) #  
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

X_li_eval = list()
for(i in 1:total_num_train){
  if(i <= train_class1){
    X_li_eval[[i]] = generate_AR1_class1(n, 1)
  } else{
    X_li_eval[[i]] = generate_AR1_class2(n, 1)
  }
}
df_train <- do.call(rbind, lapply(X_li_eval, function(x) data.frame(t(x))))
df_train = t(as.matrix(df_train))

X_li_test = list()
for(i in 1:total_num_test){
  if(i <= 25){
    X_li_test[[i]] = generate_AR1_class1(n, 1)
  } else{
    X_li_test[[i]] = generate_AR1_class2(n, 1)
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
prob_test = predict(lfreg.model, newdata = newdata, type = "probabilities")

feature_flrg = as.vector(log(prob_test/(1-prob_test)))
print(feature_flrg)

