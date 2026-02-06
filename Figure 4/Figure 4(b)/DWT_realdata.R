#  classfication for EEG 
library(wavelets)
library(MASS)
library(sparsediscrim)

n = 4096
k = 2
N = n/k 
J = 5
n_train_1 = 252
n_train_2 = 252
n_train_total = n_train_1 + n_train_2

n_test_1 = 75
n_test_2 = 75
n_test_total = n_test_1 + n_test_2
Y = factor(c(rep(1, n_train_1), rep(2, n_train_2)))
Y_test = factor(c(rep(1, n_test_1), rep(2, n_test_2)))


u_dsample <- function(data, target_size = 4096) {
  n <- length(data)
  if (n <= target_size) {
    stop("Input data must have more than 4096 points.")
  }
  
  indices <- round(seq(1, n, length.out = target_size))
  return(data[indices])
}


# C3 train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_C3.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_C3.rds")

for(j in 1:504){
  X_li_train[[j]] = u_dsample(X_li_train[[j]])
}

for(j in 1:150){
  X_li_test[[j]] = u_dsample(X_li_test[[j]])
}

# classification 

X = rep(0, J*k)
X_test = rep(0, J*k)
for(i in 1:n_train_total){
  if(i <= n_train_1){
    ts = X_li_train[[i]]
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
    ts = X_li_train[[i]]
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
    ts = X_li_test[[i]]
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
    ts = X_li_test[[i]]
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
y_hat_c3 = predict(DWT, df_test)
print(paste("Channel C3 accuracy:", round(sum(predict(DWT, df_test) == Y_test)/n_test_total, 3)))


# C4 train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_C4.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_C4.rds")

for(j in 1:504){
  X_li_train[[j]] = u_dsample(X_li_train[[j]])
}

for(j in 1:150){
  X_li_test[[j]] = u_dsample(X_li_test[[j]])
}

# classification  
X = rep(0, J*k)
X_test = rep(0, J*k)
for(i in 1:n_train_total){
  if(i <= n_train_1){
    ts = X_li_train[[i]]
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
    ts = X_li_train[[i]]
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
    ts = X_li_test[[i]]
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
    ts = X_li_test[[i]]
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
y_hat_c4 = predict(DWT, df_test)
print(paste("Channel C4 accuracy:", round(sum(predict(DWT, df_test) == Y_test)/n_test_total, 3)))



# CZ train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_CZ.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_CZ.rds")

for(j in 1:504){
  X_li_train[[j]] = u_dsample(X_li_train[[j]])
}

for(j in 1:150){
  X_li_test[[j]] = u_dsample(X_li_test[[j]])
}
# classification  
X = rep(0, J*k)
X_test = rep(0, J*k)
for(i in 1:n_train_total){
  if(i <= n_train_1){
    ts = X_li_train[[i]]
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
    ts = X_li_train[[i]]
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
    ts = X_li_test[[i]]
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
    ts = X_li_test[[i]]
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
y_hat_cz = predict(DWT, df_test)
print(paste("Channel CZ accuracy:", round(sum(predict(DWT, df_test) == Y_test)/n_test_total, 3)))


# Majority voting 

preds_matrix <- cbind(as.numeric(y_hat_cz), 
                      as.numeric(y_hat_c3), 
                      as.numeric(y_hat_c4))

# 2. Perform Majority Voting
# Since your levels are 1 and 2:
# We count how many times '1' appears in each row.
# If '1' appears 2 or more times, the winner is 1. Otherwise, it is 2.
votes_for_1 <- rowSums(preds_matrix == 1)
majority <- as.factor(ifelse(votes_for_1 >= 2, 1, 2))

print(paste("Majority voting accuracy:", round(sum(majority == Y_test)/150, 3)))


