library(wavethresh)

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
    
    print(system.time({train_class1 = 252
    train_class2 = 252
    total_num_train = train_class1 + train_class2
    n = length(X_li_train[[1]])
    print(n)
    X_li_s = list()
    for(i in 1:total_num_train){
      X_li_s[[i]] = ewspec(X_li_train[[i]])
    }
    
    
    J = log(n, 2) - 1
    
    S_1j = list()
    for(j in 0:J){
      aux = rep(0, n)
      for(i in 1:train_class1){
        aux = aux + accessD(X_li_s[[i]]$S, level = j)
      }
      
      S_1j[[j+1]] = aux/train_class1
    }
    
    S_2j = list()
    for(j in 0:J){
      aux = rep(0, n)
      for(i in (train_class1+1):total_num_train){
        aux = aux + accessD(X_li_s[[i]]$S, level = j)
      }
      
      S_2j[[j+1]] = aux/train_class2
    }
    
    sigma_jk = list()
    for(j in 0:J){
      aux = rep(0, n)
      for(i in 1:train_class1){
        aux = aux + (accessD(X_li_s[[i]]$S, level = j) - S_1j[[j+1]])^2
      }
      for(i in (train_class1+1):total_num_train){
        aux = aux + (accessD(X_li_s[[i]]$S, level = j) - S_2j[[j+1]])^2
      }
      sigma_jk[[j+1]] = aux/total_num_train
    }
    
    # choose M
    delta_jk = list()
    for(j in 0:J){
      delta_jk[[j+1]] = (S_1j[[j+1]] - S_2j[[j+1]])^2 / sigma_jk[[j+1]]
    }
    
    all_values <- unlist(delta_jk, use.names = FALSE)
    
    top_n <- 0.04 * length(all_values)
    top_indices <- order(all_values, decreasing = TRUE)[1:top_n]
    top_values <- all_values[top_indices]
    
    get_positions <- function(idx) {
      list_id <- ((idx - 1) %/% n) + 1
      index_in_list_element <- ((idx - 1) %% n) + 1
      c(list_id = list_id, position = index_in_list_element)
    }
    
    positions <- t(sapply(top_indices, get_positions))
    
    result <- data.frame(
      value = top_values,
      list_id = positions[, "list_id"],
      position_in_list_element = positions[, "position"]
    )
    
    n_test = length(X_li_test)
    y_pre = c()
    for(i in 1:n_test){
      if(i <= n_test/2){
        ts_classi =  X_li_test[[i]]
        s_classi = ewspec(ts_classi)
        D_1 = 0
        D_2 = 0
        for(num in 1:top_n){
          j = result$list_id[num]
          k = result$position_in_list_element[num]
          D_1 = D_1 + (accessD(s_classi$S, level = j - 1)[k] - S_1j[[j]][k])^2/sigma_jk[[j]][k]
          D_2 = D_2 + (accessD(s_classi$S, level = j - 1)[k] - S_2j[[j]][k])^2/sigma_jk[[j]][k]
        }
        
        if(D_1 <= D_2){
          y_pre[i] = 0
        } else{
          y_pre[i] = 1
        }
      } else{
        ts_classi = X_li_test[[i]]
        s_classi = ewspec(ts_classi)
        D_1 = 0
        D_2 = 0
        for(num in 1:top_n){
          j = result$list_id[num]
          k = result$position_in_list_element[num]
          D_1 = D_1 + (accessD(s_classi$S, level = j - 1)[k] - S_1j[[j]][k])^2/sigma_jk[[j]][k]
          D_2 = D_2 + (accessD(s_classi$S, level = j - 1)[k] - S_2j[[j]][k])^2/sigma_jk[[j]][k]
        }
        if(D_1 <= D_2){
          y_pre[i] = 0
        } else{
          y_pre[i] = 1
        }
      }
    }
    
    pre_res = (sum(y_pre[1:(n_test/2)] == 0) + sum(y_pre[(n_test/2+1):n_test] == 1))/n_test
      
    }))
    
  }
}