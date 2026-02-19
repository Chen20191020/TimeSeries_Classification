
library(fda)
library(refund)

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
      Y = as.factor(c(rep(1, 252), rep(0, 252)))
      Y_test = as.factor(c(rep(1, 75), rep(0, 75)))
      basis_num = 5
      X_li_train_trancate = list()
      n_select <- min(unlist(lapply(X_li_train, length)))
      for(j in 1:504){
        total_time_points <- length(X_li_train[[j]])
        step <- total_time_points / n_select
        selected_indices <- round(seq(1, total_time_points, length.out = n_select))
        X_li_train_trancate[[j]] = X_li_train[[j]][selected_indices]
      }
      
      df_train <- do.call(rbind, lapply(X_li_train_trancate, function(x) data.frame(t(x))))
      
      X_li_test_trancate = list()
      for(j in 1:150){
        total_time_points <- length(X_li_test[[j]])
        step <- total_time_points / n_select
        selected_indices <- round(seq(1, total_time_points, length.out = n_select))
        X_li_test_trancate[[j]] = X_li_test[[j]][selected_indices]
      }
      df_test <- do.call(rbind, lapply(X_li_test_trancate, function(x) data.frame(t(x))))
      
      
      df_test = t(as.matrix(df_test))
      df_train = t(as.matrix(df_train))
      
      x = df_train
      xbasis = create.bspline.basis(c(1,n_select), basis_num) # 5 basis functions
      xfd = smooth.basis(c(1:n_select),x, xbasis)$fd
      bbasis = create.bspline.basis(c(0,n_select), basis_num)
      betalist = list(bbasis)
      formula = Y ~ xfd
      lfreg.model = lfreg(formula, betalist = betalist)
      
      # Prediction on new data
      newdata = list(df_test)
      # newdata = list(xfd_1, latitude, longitude)
      yhat = predict(lfreg.model, newdata = newdata, type = "labels")
      
      logistic_reg = sum(as.vector(yhat) == Y_test)/150
    }))

  }
}