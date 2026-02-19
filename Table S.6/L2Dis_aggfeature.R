

# Algo

sim_func_auto_faster_L2feature = function(n, c_setting, c_1, b_1, c_2, b_2, train_class1, train_class2, test_class1, test_class2){
  # using auto.select 
  total_num_test = test_class1 + test_class2
  total_num_train = train_class1 + train_class2
  X_li_eval = list()
  #ma_diff = c()
  #para_c_b = list()
  
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
  
  coeffi_list_class1 = coeffi_list[1:train_class1]
  coeffi_list_class2 = coeffi_list[(1+train_class1):train_class2]
  
  max_len = max(lengths(coeffi_list_class1))
  coeffi_list_mean_class1 = lapply(1:max_len, function(i) {
    vecs = lapply(coeffi_list_class1, function(x) if (i <= length(x)) x[[i]] else NULL)
    vecs = Filter(Negate(is.null), vecs)
    Reduce(`+`, vecs) / length(vecs)
  })
  
  
  max_len = max(lengths(coeffi_list_class2))
  coeffi_list_mean_class2 = lapply(1:max_len, function(i) {
    vecs = lapply(coeffi_list_class2, function(x) if (i <= length(x)) x[[i]] else NULL)
    vecs = Filter(Negate(is.null), vecs)
    Reduce(`+`, vecs) / length(vecs)
  })
  
  
  tilde_b = min(b_x, b_y)
  coeff_avg_class1 = coeffi_list_mean_class1[[tilde_b]]
  coeff_avg_class2 = coeffi_list_mean_class2[[tilde_b]]
  
  # test set 
  
  X_li_test = list()
  
  for(i in 1:total_num_test){
    if(i <= total_num_test/2){
      X_li_test[[i]] = get(function_names[2*c_setting-1])(n, 1)
    } else{
      X_li_test[[i]] = get(function_names[2*c_setting])(n, 1)
    }
  }
  
  #ma_diff_test = c()
  #b_z = c()
  coeffi_list = list()
  
  for(num_obs in 1:total_num_test){
    res = auto.fit.legen_fast(X_li_test[[num_obs]], c_1, b_1, m = 500, method = "LOOCV", inte = FALSE)
    #b_z[num_obs] = 
    coeffi_list[[num_obs]] = res$Estimate[[res$BC[[2]]]]
  }
  
  #b_star_z = min(b_z)
  
  Y = c(rep(1, test_class1), rep(2, test_class2))
  result = c()
  y_pre = rep(0, total_num_test)
  
  for(num_obs in 1:total_num_test){
    L2_i_class1 = sum((coeffi_list[[num_obs]] - coeff_avg_class1)^2/500)
    L2_i_class2 = sum((coeffi_list[[num_obs]] - coeff_avg_class2)^2/500)
    
    y_pre[num_obs] = ifelse(L2_i_class1 <= L2_i_class2, 1, 2)
    
  }
  result = sum(y_pre == Y)
  return(result/total_num_test)
}


# Model 1,2,a,b,c
model_name = c(1,2,"a","b","c")

for(case in 1:5){
  print(paste("Model", model_name[case]))
  res = c()
  for(j in 1:500){
    res[j] = sim_func_auto_faster_L2feature(1600, case, 1:5, 1:4, 1:5, 1:4, 100, 100, 25, 25)
  }
  print(round(mean(res),2))
  print(round(sd(res), 2))
  
}

# Real Data 

sim_func_auto_faster_L2Dis_realdata = function(Train_list, Testing_list){
  # using auto.select 
  test_class1 = length(Testing_list)/2
  test_class2 = length(Testing_list)/2 
  train_class1 = length(Train_list)/2
  train_class2 = length(Train_list)/2
  
  total_num_test = test_class1 + test_class2
  total_num_train = train_class1 + train_class2
  
  b_x = c()
  b_y = c()
  coeffi_list = list()
  
  for(num_obs in 1:total_num_train){
    if(num_obs <= train_class1){
      
      res = auto.fit.legen_fast(Train_list[[num_obs]], 1:10, 1:8, m = 500, method = "LOOCV")
      b_x[num_obs] = res$BC[[2]]
      coeffi_list[[num_obs]] = res$Estimate[[1]]
      
    } else{
      res = auto.fit.legen_fast(Train_list[[num_obs]], 1:10, 1:8, m = 500, method = "LOOCV") 
      b_y[(num_obs-train_class1)] = res$BC[[2]]
      coeffi_list[[num_obs]] = res$Estimate[[1]]
      
    }
  }
  
  #b_star_x = min(b_x)
  #b_star_y = min(b_y)
  coeffi_list_class1 = coeffi_list[1:train_class1]
  coeffi_list_class2 = coeffi_list[(1+train_class1):train_class2]
  
  max_len = max(lengths(coeffi_list_class1))
  coeffi_list_mean_class1 = lapply(1:max_len, function(i) {
    vecs = lapply(coeffi_list_class1, function(x) if (i <= length(x)) x[[i]] else NULL)
    vecs = Filter(Negate(is.null), vecs)
    Reduce(`+`, vecs) / length(vecs)
  })
  
  
  max_len = max(lengths(coeffi_list_class2))
  coeffi_list_mean_class2 = lapply(1:max_len, function(i) {
    vecs = lapply(coeffi_list_class2, function(x) if (i <= length(x)) x[[i]] else NULL)
    vecs = Filter(Negate(is.null), vecs)
    Reduce(`+`, vecs) / length(vecs)
  })
  
  
  tilde_b = min(b_x, b_y)
  coeff_avg_class1 = coeffi_list_mean_class1[[tilde_b]]
  coeff_avg_class2 = coeffi_list_mean_class2[[tilde_b]]
  
  #coeff_avg_class1 = Reduce(`+`, coeffi_list[1:train_class1])/train_class1
  
  #coeff_avg_class2 = Reduce(`+`, coeffi_list[(train_class1+1):total_num_train])/train_class2
  #ma_diff_test = c()
  #b_z = c()
  coeffi_list = list()
  
  for(num_obs in 1:total_num_test){
    res = auto.fit.legen_fast(Testing_list[[num_obs]], 1:10, 1:8, m = 500, method = "LOOCV")
    #b_z[num_obs] = res$BC[[2]]
    coeffi_list[[num_obs]] = res$Estimate[[res$BC[[2]]]]
  }
  
  #b_star_z = min(b_z)
  
  Y = c(rep(1, test_class1), rep(2, test_class2))
  result = c()
  y_pre = rep(0, total_num_test)
  
  for(num_obs in 1:total_num_test){
    L2_i_class1 = sum((coeffi_list[[num_obs]] - coeff_avg_class1)^2/500)
    L2_i_class2 = sum((coeffi_list[[num_obs]] - coeff_avg_class2)^2/500)
    
    y_pre[num_obs] = ifelse(L2_i_class1 < L2_i_class2, 1, 2)
    
  }
  result = sum(y_pre == Y)
  return(result/total_num_test)
}


# Real Data 

# CZ train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_CZ.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_CZ.rds")

# classification 
print("Channel CZ accuracy:")
print(sim_func_auto_faster_L2Dis_realdata(X_li_train, X_li_test))




