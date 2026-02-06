
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
    aux_train <- aux_train[, -1]        # Remove first col
    mat_train <- as.matrix(aux_train)
    X_li_train <- split(mat_train, row(mat_train))
    
    # 2. Test Data
    aux_test  <- read.csv(file_test)
    aux_test  <- aux_test[, -1]         # Remove first col
    mat_test  <- as.matrix(aux_test)
    X_li_test <- split(mat_test, row(mat_test))
    
    # --- Run Your Algorithm ---
    # The 'system.time' will measure how long this specific dataset takes
    print("Starting Algorithm...")
    
    print(system.time({
      ma_diff_train = c()
      
      bc_list_train_fast = list()
      bc_list_test_fast = list()
      
      for(num_obs in 1:504){
        res = auto.fit.legen_fast(X_li_train[[num_obs]], c(1:10), c(1:4), method = "LOOCV", inte = T)
        bc_list_train_fast[[num_obs]] = res[[4]]
        res_coef = res[[1]]
        phi_j = res_coef[[length(res_coef)]]
        ma_diff_train[num_obs] = max(phi_j) - min(phi_j)
      }
      
      Y = c(rep("abnormal", 504/2), rep("normal", 504/2))
      diff = seq(1/2*min(ma_diff_train), 2*max(ma_diff_train), length.out = 10000)
      result = c()
      for(j in 1:10000){
        y_pre = rep(0, 504)
        for(num_obs in 1:504){
          y_pre[num_obs] = ifelse(ma_diff_train[num_obs] > diff[j], "abnormal", "normal")
        }
        result[j] = sum(y_pre == Y)
      }
      threshold = round(diff[which.max(result)], 2)
      # test
      ma_diff = c()
      
      for(num_obs in 1:length(X_li_test)){
        res = auto.fit.legen_fast(X_li_test[[num_obs]], c(1:10), c(1:3), method = "LOOCV", inte = T)
        bc_list_test_fast[[num_obs]] = res[[4]]
        res_coef = res[[1]]
        phi_j = res_coef[[length(res_coef)]]
        ma_diff[num_obs] = max(phi_j) - min(phi_j)
      }
      
      result = c()
      Y = c(rep("abnormal", length(X_li_test)/2), rep("normal", length(X_li_test)/2))
      y_pre = rep(0, length(X_li_test))
      for(num_obs in 1:length(X_li_test)){
        y_pre[num_obs] = ifelse(ma_diff[num_obs] > threshold, "abnormal", "normal")
      }
      result = sum(y_pre == Y)/length(X_li_test)
    }))
    
  }
}