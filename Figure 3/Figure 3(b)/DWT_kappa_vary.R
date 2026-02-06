library(wavelets)
library(MASS)

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


n = 1024
k = 2
N = n/k 
J = 5 
n_train_1 = 50

n_test_1 = 25
n_test_2 = 25
n_test_total = n_test_1 + n_test_2
Y_test = factor(c(rep(1, n_test_1), rep(2, n_test_2)))
n_del = seq(1, 5, by = 0.5)
del = 0.2
mean_ac = c()
sd_ac = c()


for(num_delta in 1:length(n_del)){
  n_train_2 = n_del[num_delta]*n_train_1
  n_train_total = n_train_1 + n_train_2
  Y = factor(c(rep(1, n_train_1), rep(2, n_train_2)))
  res_dwt = c()
  for(num_res in 1:500){
    X = rep(0, J*k)
    X_test = rep(0, J*k)
    for(i in 1:n_train_total){
      if(i <= n_train_1){
        ts = generate_AR1_test_class1(n, 1, del)
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
        ts = generate_AR1_test_class2(n, 1, del)
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
    # DWT <- lda(X, Y)
    
    for(i in 1:n_test_total){
      if(i <= n_test_1){
        ts = generate_AR1_test_class1(n, 1, del)
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
        ts = generate_AR1_test_class2(n, 1, del)
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
  mean_ac[num_delta] = mean(res_dwt)
  sd_ac[num_delta] = sd(res_dwt)
}

# The accuracy for kappa values from 1 to 5, in order.
print(round(mean_ac, 3))
print(round(sd_ac, 3))

