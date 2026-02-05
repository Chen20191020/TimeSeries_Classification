
# Installed the packages first 
#install.packages("wavelets")
#install.packages("sparsediscrim")

library(wavelets)
library(MASS)
library(sparsediscrim)


n = 1024
k = 2
N = n/k 
J = 5
n_train_1 = 100
n_train_2 = 100
n_train_total = n_train_1 + n_train_2

n_test_1 = 25
n_test_2 = 25
n_test_total = n_test_1 + n_test_2
Y = factor(c(rep(1, n_train_1), rep(2, n_train_2)))
Y_test = factor(c(rep(1, n_test_1), rep(2, n_test_2)))


for(c_setting in 1:3){
  res_dwt = c()
  print(paste("Model", c_setting+6))
  for(num_res in 1:500){
    X = rep(0, J*k)
    X_test = rep(0, J*k)
    for(i in 1:n_train_total){
      if(i <= n_train_1){
        ts = get(function_names[2*c_setting - 1])(n, 1)
        W_1 = wavelets::modwt(ts[1:(n/k)], n.levels = J)
        W_2 = wavelets::modwt(ts[(n/k + 1):n], n.levels = J) 
        L_j = W_1@n.boundary
        W_1 = W_1@W
        W_2 = W_2@W
        V_1 = c()
        V_2 = c()
        for(j in 1:J){
          M = N - L_j[j] + 1
          V_1[j] = sum((as.vector(W_1[[j]])[(L_j[j]-1):(N-1)])^2)/M
          V_2[j] = sum((as.vector(W_2[[j]])[(L_j[j]-1):(N-1)])^2)/M
        }
        X = rbind(X, c(V_1, V_2))
      } else{
        ts = get(function_names[2*c_setting])(n, 1)
        W_1 = wavelets::modwt(ts[1:(n/k)], n.levels = J)
        W_2 = wavelets::modwt(ts[(n/k + 1):n], n.levels = J) 
        L_j = W_1@n.boundary
        W_1 = W_1@W
        W_2 = W_2@W
        V_1 = c()
        V_2 = c()
        for(j in 1:J){
          M = N - L_j[j] + 1
          V_1[j] = sum((as.vector(W_1[[j]])[(L_j[j]-1):(N-1)])^2)/M
          V_2[j] = sum((as.vector(W_2[[j]])[(L_j[j]-1):(N-1)])^2)/M
        }
        X = rbind(X, c(V_1, V_2))
      }
    }
    
    X = X[-1,]
    
    df = cbind(X, Y)
    df = as.data.frame(df)
    df$Y = as.factor(df$Y)
    DWT = qda_diag(Y~., df)
    
    for(i in 1:n_test_total){
      if(i <= n_test_1){
        ts = get(function_names[2*c_setting - 1])(n, 1)
        W_1 = wavelets::modwt(ts[1:(n/k)], n.levels = J)
        W_2 = wavelets::modwt(ts[(n/k + 1):n], n.levels = J) 
        L_j = W_1@n.boundary
        W_1 = W_1@W
        W_2 = W_2@W
        V_1 = c()
        V_2 = c()
        for(j in 1:J){
          M = N - L_j[j] + 1
          V_1[j] = sum((as.vector(W_1[[j]])[(L_j[j]-1):(N-1)])^2)/M
          V_2[j] = sum((as.vector(W_2[[j]])[(L_j[j]-1):(N-1)])^2)/M
        }
        X_test = rbind(X_test, c(V_1, V_2))
      } else{
        ts = get(function_names[2*c_setting])(n, 1)
        W_1 = wavelets::modwt(ts[1:(n/k)], n.levels = J)
        W_2 = wavelets::modwt(ts[(n/k + 1):n], n.levels = J) 
        L_j = W_1@n.boundary
        W_1 = W_1@W
        W_2 = W_2@W
        V_1 = c()
        V_2 = c()
        for(j in 1:J){
          M = N - L_j[j] + 1
          V_1[j] = sum((as.vector(W_1[[j]])[(L_j[j]-1):(N-1)])^2)/M
          V_2[j] = sum((as.vector(W_2[[j]])[(L_j[j]-1):(N-1)])^2)/M
        }
        X_test = rbind(X_test, c(V_1, V_2))
      }
    }
    X_test = X_test[-1,]
    df_test = as.data.frame(X_test)
    res_dwt[num_res] = sum(predict(DWT, df_test) == Y_test)/n_test_total
    
  }
  print(c(mean(res_dwt), sd(res_dwt)))
}
