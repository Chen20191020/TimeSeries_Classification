# -----------------------------------------------------------------------------
# 1. Helper: Fast Legendre Basis Evaluation (Unchanged)
# -----------------------------------------------------------------------------
get_legendre_matrix <- function(x, c) {
  n_obs <- length(x)
  L <- matrix(0, nrow = n_obs, ncol = c)
  
  xi <- 2 * x - 1
  
  L[, 1] <- 1
  if (c > 1) {
    L[, 2] <- xi
  }
  
  if (c > 2) {
    for (k in 3:c) {
      degree <- k - 1
      L[, k] <- ( (2 * degree - 1) * xi * L[, k - 1] - (degree - 1) * L[, k - 2] ) / degree
    }
  }
  
  scales <- sqrt(2 * (0:(c - 1)) + 1)
  L <- sweep(L, 2, scales, "*")
  
  return(L)
}

# -----------------------------------------------------------------------------
# 2. Optimized Estimation Function (beta_l) -- UPDATED
# -----------------------------------------------------------------------------
beta_l_fast <- function(ts, c, b, inte = TRUE) {
  n <- length(ts)
  
  # Indices for the regression (from b+1 to n)
  idx <- (b + 1):n
  n_reg <- length(idx)
  
  # 1. Response Vector X
  X <- matrix(ts[idx], ncol = 1)
  
  # 2. Pre-compute Legendre Basis
  time_points <- idx / n
  B <- get_legendre_matrix(time_points, c)
  
  # 3. Construct Design Matrix Y efficiently
  # We construct the "lag matrix" first: [1, Xt-1, Xt-2 ... Xt-b]
  
  # Create Lags
  lags_mat <- matrix(NA, nrow = n_reg, ncol = b)
  for(j in 1:b){
    lags_mat[, j] <- ts[idx - j]
  }
  
  # Add Intercept column if requested
  if(inte){
    design_base <- cbind(1, lags_mat)
    num_predictors <- b + 1
  } else {
    design_base <- lags_mat
    num_predictors <- b
  }
  
  # 4. Create the Full TV-AR Design Matrix
  # Structure: [ Basis | Basis*Lag1 | ... ] if intercept=TRUE
  # Structure: [ Basis*Lag1 | Basis*Lag2 | ... ] if intercept=FALSE
  
  Y <- matrix(0, nrow = n_reg, ncol = c * num_predictors)
  
  for (j in 1:num_predictors) {
    col_start <- (j - 1) * c + 1
    col_end   <- j * c
    
    # Multiply Basis B by the predictor column (element-wise for each row)
    Y[, col_start:col_end] <- B * design_base[, j]
  }
  
  # 5. Solve for Beta
  XtY <- crossprod(Y, X)
  YtY <- crossprod(Y)
  
  # Add small ridge for stability if needed
  beta <- solve(YtY, XtY, tol = 1e-40)
  
  return(list(beta, Y))
}

# -----------------------------------------------------------------------------
# 3. Optimized Curve Reconstruction (alpha.legen) -- UPDATED
# -----------------------------------------------------------------------------
alpha.legen_fast <- function(ts, c, b, m = 500, inte = TRUE) {
  
  # 1. Estimate Beta
  est <- beta_l_fast(ts, c, b, inte)
  beta_vec <- est[[1]]
  
  # 2. Generate Grid and Basis Matrix
  grid_points <- (1:m) / m
  B_grid <- get_legendre_matrix(grid_points, c) 
  
  # 3. Reconstruct curves
  # Number of parameter curves: b+1 (with intercept) or b (without)
  num_curves <- if(inte) b + 1 else b
  
  # Reshape beta into a matrix: (c) rows x (num_curves) columns
  beta_mat <- matrix(beta_vec, nrow = c, ncol = num_curves)
  
  # Calculate curves
  curves <- B_grid %*% beta_mat
  
  # 4. Convert to list format
  l_alpha <- vector("list", num_curves)
  for (j in 1:num_curves) {
    l_alpha[[j]] <- curves[, j]
  }
  
  return(l_alpha)
}



# -----------------------------------------------------------------------------
# 4. Optimized Fit & Residual Function -- UPDATED
# -----------------------------------------------------------------------------
fix.fit.legen_fast <- function(ts, c, b, m, inte = TRUE){
  
  # 1. Perform Estimation
  est_results <- beta_l_fast(ts, c, b, inte)
  beta_hat <- est_results[[1]]
  Y <- est_results[[2]]
  
  # 2. Calculate Residuals
  n <- length(ts)
  observed_y <- ts[(b+1):n]
  fitted_vals <- Y %*% beta_hat
  residuals <- observed_y - fitted_vals
  
  # 3. Reconstruct Curves
  grid_points <- (1:m) / m
  B_grid <- get_legendre_matrix(grid_points, c)
  
  num_curves <- if(inte) b + 1 else b
  beta_mat <- matrix(beta_hat, nrow = c, ncol = num_curves)
  curves_mat <- B_grid %*% beta_mat
  
  ts_coef_list <- vector("list", num_curves)
  for(j in 1:num_curves){
    ts_coef_list[[j]] <- curves_mat[, j]
  }
  
  return(list(
    ols.coef = beta_hat, 
    ts.coef = ts_coef_list, 
    Residuals = as.numeric(residuals),
    IncludesIntercept = inte
  ))
}

# -----------------------------------------------------------------------------
# 5. Optimized Leave-One-Out Cross-Validation -- UPDATED
# -----------------------------------------------------------------------------
alpha.loocv.l_fast <- function(ts, c, b, inte = TRUE){
  
  # 1. Get Fit
  est <- beta_l_fast(ts, c, b, inte)
  beta_hat <- est[[1]]
  Y <- est[[2]]
  
  n_reg <- nrow(Y)
  
  # 2. Calculate Residuals
  y_true <- ts[(b+1):length(ts)]
  residuals <- y_true - (Y %*% beta_hat)
  
  # 3. Calculate Diagonal Hat Values
  XtX <- crossprod(Y)
  inv_XtX <- solve(XtX, tol = 1e-40)
  
  H_part <- Y %*% inv_XtX 
  h_diag <- rowSums(H_part * Y)
  
  # 4. PRESS Statistic
  press_errors <- (residuals / (1 - h_diag))^2
  loocv_score <- sum(press_errors) / n_reg
  
  return(c(c, b, loocv_score))
}

# -----------------------------------------------------------------------------
# 6. Optimized Auto-Fit & Model Selection -- UPDATED
# -----------------------------------------------------------------------------
auto.fit.legen_fast = function(ts, candi_c, candi_b, m = 500, method = "CV", threshold = 0, inte = TRUE){
  
  # 1. Create Parameter Grid
  param_grid <- expand.grid(b = candi_b, c = candi_c)
  n_runs <- nrow(param_grid)
  
  res.bc <- matrix(NA, nrow = n_runs, ncol = 3)
  colnames(res.bc) <- c("c", "b", "cv")
  
  # 2. Iterate over Grid
  for(k in 1:n_runs){
    curr_c <- param_grid$c[k]
    curr_b <- param_grid$b[k]
    
    if(method == "CV"){
      # Uses the fast CV function (updated in previous answers)
      # Ensure you have defined alpha.cv.l_fast correctly with inte
      res.bc[k, ] <- alpha.cv.l_fast(ts, curr_c, curr_b, inte)
    } else {
      res.bc[k, ] <- alpha.loocv.l_fast(ts, curr_c, curr_b, inte)
    }
  }
  
  # 3. Model Selection Logic
  min_idx <- which.min(res.bc[, "cv"])
  b.s <- res.bc[min_idx, "b"]
  
  subset_res <- res.bc[res.bc[, "b"] == b.s, , drop = FALSE]
  subset_res <- subset_res[order(subset_res[, "c"]), , drop = FALSE]
  
  if(method == "Elbow"){
    cv_scores <- subset_res[, "cv"]
    n_scores <- length(cv_scores)
    
    if(n_scores < 2){
      c.s <- subset_res[1, "c"]
    } else {
      ratios <- abs(cv_scores[1:(n_scores-1)] / cv_scores[2:n_scores] - 1)
      
      if(threshold == 0){
        idx_max <- which.max(ratios)
        c.s <- subset_res[idx_max + 1, "c"]
      } else {
        valid_improvements <- which(ratios >= threshold)
        if(length(valid_improvements) > 0){
          c.s <- subset_res[max(valid_improvements) + 1, "c"]
        } else {
          c.s <- subset_res[1, "c"]
        }
      }
    }
  } else {
    c.s <- res.bc[min_idx, "c"]
  }
  
  # 4. Final Estimate
  estimate <- alpha.legen_fast(ts, c.s, b.s, m, inte)
  final_betas <- beta_l_fast(ts, c.s, b.s, inte)[[1]]
  
  return(list(
    Estimate = estimate, 
    CV = res.bc, 
    Coefficients = final_betas, 
    BC = c(c.s, b.s)
  ))
}



# -----------------------------------------------------------------------------
# 6.function for simulations
# -----------------------------------------------------------------------------

sim_func_without_auto_faster = function(n, c_setting, c_1, b_1, c_2, b_2, train_class1, train_class2, test_class1, test_class2){
  # using auto.select 
  total_num_test = test_class1 + test_class2
  total_num_train = train_class1 + train_class2
  X_li_eval = list()
  ma_diff = c()
  para_c_b = list()
  
  for(i in 1:total_num_train){
    if(i <= train_class1){
      X_li_eval[[i]] = get(function_names[2*c_setting-1])(n, 1)
    } else{
      X_li_eval[[i]] = get(function_names[2*c_setting])(n, 1)
    }
  }
  
  coeffi_list = list()
  
  for(num_obs in 1:total_num_train){
    if(num_obs <= train_class1){
      res =  fix.fit.legen_fast(X_li_eval[[num_obs]], c_1, b_1, 500, inte = FALSE)
      coeffi_list[[num_obs]] = res$ts.coef
      
    } else{
      res = fix.fit.legen_fast(X_li_eval[[num_obs]], c_2, b_2, 500, inte = FALSE) 
      coeffi_list[[num_obs]] = res$ts.coef
    }
    
  }
  
  b_star_x = min(b_1)
  b_star_y = min(b_2)
  
  for(num_obs in 1:total_num_train){
    if(num_obs <= train_class1){
      b_end = b_1
      b_start = max(b_end - b_star_x + 1, b_star_x)
      
      D_j_list = unlist(lapply(coeffi_list[[num_obs]], function(x) max(x) - min(x))) 
      ma_diff[num_obs]  = max(D_j_list[b_start:b_end])
    } else{
      b_end = b_2
      b_start = max(b_end - b_star_y + 1, b_star_y)
      D_j_list = unlist(lapply(coeffi_list[[num_obs]], function(x) max(x) - min(x))) 
      ma_diff[num_obs]  = max(D_j_list[b_start:b_end])
    }
  }
  
  med_class1 = median(ma_diff[1:train_class1])
  med_class2 = median(ma_diff[(1+train_class1):total_num_train])
  Y = c(rep(1, train_class1), rep(2, train_class2))
  
  diff = seq(min(ma_diff)/2, 2*max(ma_diff), length.out = 5000)
  result = c()
  
  for(j in 1:5000){
    y_pre = rep(0, length(ma_diff))
    for(num_obs in 1:length(ma_diff)){
      if(med_class1 < med_class2){
        y_pre[num_obs] = ifelse(ma_diff[num_obs] < diff[j], 1, 2)
      } else{
        y_pre[num_obs] = ifelse(ma_diff[num_obs] >= diff[j], 1, 2)
      }
    }
    result[j] = sum(y_pre == Y)
  }
  
  threshold = diff[which.max(result)]
  
  if(length(threshold) > 1){
    threshold = threshold[1]
  }
  
  # test set 
  
  X_li_test = list()
  
  for(i in 1:total_num_test){
    if(i <= total_num_test/2){
      X_li_test[[i]] = get(function_names[2*c_setting-1])(n, 1)
    } else{
      X_li_test[[i]] = get(function_names[2*c_setting])(n, 1)
    }
  }
  
  ma_diff_test = c()
  coeffi_list = list()
  
  
  for(num_obs in 1:total_num_test){
    
    if(num_obs <= test_class1){
      res =  fix.fit.legen_fast(X_li_test[[num_obs]], c_1, b_1, 500, inte = FALSE)
      coeffi_list[[num_obs]] = res$ts.coef
      
    } else{
      res = fix.fit.legen_fast(X_li_test[[num_obs]], c_2, b_2, 500, inte = FALSE) 
      coeffi_list[[num_obs]] = res$ts.coef
    }
  }
  
  b_star_z = min(c(b_1, b_2))
  
  for(num_obs in 1:total_num_test){
    b_end = ifelse(num_obs <= test_class1, b_1, b_2)
    b_start = max(b_end - b_star_z + 1, b_star_z)
    D_j_list = unlist(lapply(coeffi_list[[num_obs]], function(x) max(x) - min(x))) 
    ma_diff_test[num_obs]  = max(D_j_list[b_start:b_end])
    
  }
  
  Y = c(rep(1, test_class1), rep(2, test_class2))
  result = c()
  y_pre = rep(0, length(ma_diff_test))
  
  for(num_obs in 1:length(ma_diff_test)){
    if(med_class1 < med_class2){
      y_pre[num_obs] = ifelse(ma_diff_test[num_obs] < threshold, 1, 2)
    } else{
      y_pre[num_obs] = ifelse(ma_diff_test[num_obs] >= threshold, 1, 2)
    }
    
  }
  result = sum(y_pre == Y)
  return(result/total_num_test)
}

sim_func_auto_faster = function(n, c_setting, c_1, b_1, c_2, b_2, train_class1, train_class2, test_class1, test_class2){
  # using auto.select 
  total_num_test = test_class1 + test_class2
  total_num_train = train_class1 + train_class2
  X_li_eval = list()
  ma_diff = c()
  para_c_b = list()
  
  for(i in 1:total_num_train){
    if(i <= train_class1){
      X_li_eval[[i]] = get(function_names[2*c_setting-1])(n, 1)
    } else{
      X_li_eval[[i]] = get(function_names[2*c_setting])(n, 1)
    }
  }
  
  b_x = c()
  b_y = c()
  coeffi_list = list()
  
  for(num_obs in 1:total_num_train){
    if(num_obs <= train_class1){
      
      res = auto.fit.legen_fast(X_li_eval[[num_obs]], c_1, b_1, m = 500, method = "LOOCV", inte = FALSE)
      b_x[num_obs] = res$BC[[2]]
      coeffi_list[[num_obs]] = res$Estimate
      
    } else{
      res = auto.fit.legen_fast(X_li_eval[[num_obs]], c_2, b_2, m = 500, method = "LOOCV", inte = FALSE) 
      b_y[(num_obs-train_class1)] = res$BC[[2]]
      coeffi_list[[num_obs]] = res$Estimate
      
    }
  }
  
  b_star_x = min(b_x)
  b_star_y = min(b_y)
  
  for(num_obs in 1:total_num_train){
    if(num_obs <= train_class1){
      b_end = b_x[num_obs]
      b_start = max(b_end - b_star_x + 1, b_star_x)
      
      D_j_list = unlist(lapply(coeffi_list[[num_obs]], function(x) max(x) - min(x))) 
      ma_diff[num_obs]  = max(D_j_list[b_start:b_end])
    } else{
      b_end = b_y[num_obs-train_class1]
      b_start = max(b_end - b_star_y + 1, b_star_y)
      D_j_list = unlist(lapply(coeffi_list[[num_obs]], function(x) max(x) - min(x))) 
      ma_diff[num_obs]  = max(D_j_list[b_start:b_end])
    }
  }
  
  med_class1 = median(ma_diff[1:train_class1])
  med_class2 = median(ma_diff[(1+train_class1):total_num_train])
  Y = c(rep(1, train_class1), rep(2, train_class2))
  
  diff = seq(min(ma_diff)/2, 2*max(ma_diff), length.out = 5000)
  result = c()
  
  for(j in 1:5000){
    y_pre = rep(0, length(ma_diff))
    for(num_obs in 1:length(ma_diff)){
      if(med_class1 < med_class2){
        y_pre[num_obs] = ifelse(ma_diff[num_obs] < diff[j], 1, 2)
      } else{
        y_pre[num_obs] = ifelse(ma_diff[num_obs] >= diff[j], 1, 2)
      }
    }
    result[j] = sum(y_pre == Y)
  }
  
  threshold = diff[which.max(result)]
  
  if(length(threshold) > 1){
    threshold = threshold[1]
  }
  
  # test set 
  
  X_li_test = list()
  
  for(i in 1:total_num_test){
    if(i <= total_num_test/2){
      X_li_test[[i]] = get(function_names[2*c_setting-1])(n, 1)
    } else{
      X_li_test[[i]] = get(function_names[2*c_setting])(n, 1)
    }
  }
  
  ma_diff_test = c()
  b_z = c()
  coeffi_list = list()
  
  for(num_obs in 1:total_num_test){
    res = auto.fit.legen_fast(X_li_test[[num_obs]], c_1, b_1, m = 500, method = "LOOCV", inte = FALSE)
    b_z[num_obs] = res$BC[[2]]
    coeffi_list[[num_obs]] = res$Estimate
  }
  
  b_star_z = min(b_z)
  
  for(num_obs in 1:total_num_test){
    b_end = b_z[num_obs]
    b_start = max(b_end - b_star_z + 1, b_star_z)
    D_j_list = unlist(lapply(coeffi_list[[num_obs]], function(x) max(x) - min(x))) 
    ma_diff_test[num_obs]  = max(D_j_list[b_start:b_end])
    
  }
  
  Y = c(rep(1, test_class1), rep(2, test_class2))
  result = c()
  y_pre = rep(0, length(ma_diff_test))
  
  for(num_obs in 1:length(ma_diff_test)){
    if(med_class1 < med_class2){
      y_pre[num_obs] = ifelse(ma_diff_test[num_obs] < threshold, 1, 2)
    } else{
      y_pre[num_obs] = ifelse(ma_diff_test[num_obs] >= threshold, 1, 2)
    }
    
  }
  result = sum(y_pre == Y)
  return(result/total_num_test)
}







