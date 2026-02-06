library(wavelets)
library(MASS)
library(sparsediscrim)

# 1. Define the components of your file names
prefixes = c("CZ", "C3")
indices = c(1, 2, 4, 6, 8, 10, 12)

# 2. Loop through every combination
for (prefix in prefixes) {
  for (idx in indices) {
    
    # --- Construct File Names Dynamically ---
    # Example result: "CZ_1", "C3_12", etc.
    dataset_name = paste0(prefix, "_", idx)
    
    file_train = paste0("/home/chenqian/train_", dataset_name, ".csv")
    file_test  = paste0("/home/chenqian/test_", dataset_name, ".csv")
    
    print(paste("--------------------------------------------------"))
    print(paste("Running for dataset:", dataset_name))
    print(paste("Reading:", file_train))
    
    # --- Data Loading & Preprocessing ---
    # 1. Train Data
    aux_train <- read.csv(file_train)
    aux_train <- aux_train[, -1]  # Remove first col
    n <- ncol(aux_train)
    exponent <- floor(log2(n))
    target_n <- 2^exponent
    df_trimmed <- aux_train[, 1:target_n]
    
    mat_train <- as.matrix(df_trimmed)
    X_li_train <- split(mat_train, row(mat_train))
     
    # 2. Test Data
    aux_test  <- read.csv(file_test)
    aux_test  <- aux_test[, -1]         # Remove first col
    n <- ncol(aux_test)
    exponent <- floor(log2(n))
    target_n <- 2^exponent
    df_trimmed <- aux_test[, 1:target_n]
    
    mat_test  <- as.matrix(df_trimmed)
    X_li_test <- split(mat_test, row(mat_test))
    
    
    # --- Run Your Algorithm ---
    # The 'system.time' will measure how long this specific dataset takes
    print("Starting Algorithm...")
    
    print(system.time({res_dwt = c()
    n = length(X_li_train[[1]])
    print(n)
    k = 2
    N = n/k
    J = 4 #log(N, 2)
    n_train_1 = 252
    n_train_2 = 252
    n_train_total = n_train_1 + n_train_2
    
    n_test_1 = 75
    n_test_2 = 75
    n_test_total = n_test_1 + n_test_2
    Y = factor(c(rep(1, n_train_1), rep(2, n_train_2)))
    Y_test = factor(c(rep(1, n_test_1), rep(2, n_test_2)))
    
    
    X = rep(0, J*k)
    X_test = rep(0, J*k)
    for(i in 1:n_train_total){
      if(i <= n_train_1){
        ts = X_li_train[[i]]#get(function_names[2*c_setting - 1])(n, 1)
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
        ts = X_li_train[[i]] #get(function_names[2*c_setting])(n, 1)
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
        ts = X_li_test[[i]] #get(function_names[2*c_setting - 1])(n, 1)
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
        ts = X_li_test[[i]] #get(function_names[2*c_setting])(n, 1)
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
    pre_res = sum(predict(DWT, df_test) == Y_test)/n_test_total
    }))
    
  }
}