
# Installed the packages first 
#install.packages("wavelets")
#install.packages("sparsediscrim")

library(wavelets)
library(MASS)
library(sparsediscrim)

# Helper function to process a single time series into sub_n wavelet variance features
process_ts = function(ts, n, sub_n, J) {
  segment_size = n / sub_n
  all_variances = c()
  
  for (s in 1:sub_n) {
    # Define indices for the current segment
    start_idx = (s - 1) * segment_size + 1
    end_idx   = s * segment_size
    segment   = ts[start_idx:end_idx]
    
    # Perform MODWT
    W_modwt = wavelets::modwt(segment, n.levels = J)
    L_j     = W_modwt@n.boundary
    W_coeffs = W_modwt@W
    
    V_s = rep(0, J)
    N_seg = length(segment)
    
    for (j in 1:J) {
      M = N_seg - L_j[j] + 1
      # Wavelet variance calculation for the j-th level
      V_s[j] = sum((as.vector(W_coeffs[[j]])[(L_j[j]):N_seg])^2) / M
    }
    all_variances = c(all_variances, V_s)
  }
  return(all_variances)
}

n = 1024
k = 2
N = n/k 
J = 10
sub_n = 2
n_train_1 = 50
n_train_2 = 250
n_train_total = n_train_1 + n_train_2

n_test_1 = 25
n_test_2 = 25
n_test_total = n_test_1 + n_test_2
Y = factor(c(rep(1, n_train_1), rep(2, n_train_2)))
Y_test = factor(c(rep(1, n_test_1), rep(2, n_test_2)))


for(c_setting in 1:6){
  res_dwt = c()
  print(paste("Model", as.roman(c_setting)))
  for(num_res in 1:500){
    X = matrix(0, nrow = 0, ncol = J * sub_n)
    X_test = matrix(0, nrow = 0, ncol = J * sub_n)
    
    # 1. Training Loop
    for(i in 1:n_train_total){
      func_idx = if(i <= n_train_1) (2*c_setting - 1) else (2*c_setting)
      ts = get(function_names[func_idx])(n, 1)
      
      features = process_ts(ts, n, sub_n, J)
      X = rbind(X, features)
    }
    
    # 2. Model Fitting
    df = as.data.frame(cbind(X, Y))
    df$Y = as.factor(df$Y)
    DWT = lda_diag(Y ~ ., data = df)
    
    # 3. Testing Loop
    for(i in 1:n_test_total){
      func_idx = if(i <= n_test_1) (2*c_setting - 1) else (2*c_setting)
      ts = get(function_names[func_idx])(n, 1)
      
      features = process_ts(ts, n, sub_n, J)
      X_test = rbind(X_test, features)
    }
    
    # Ensure X_test is a data frame and column names match the training set
    df_test = as.data.frame(X_test)
    colnames(df_test) = colnames(df)[-ncol(df)] 
    
    # Perform prediction using the fitted DWT model
    predictions = predict(DWT, df_test)
    
    # Calculate accuracy
    res_dwt[num_res] = sum(predictions == Y_test) / n_test_total
  }
  
  print(c(mean(res_dwt), sd(res_dwt)))
}
