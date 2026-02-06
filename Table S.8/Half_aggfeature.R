
# Algo
sim_func_auto_faster_featurehalf = function(n, c_setting, c_1, b_1, c_2, b_2, train_class1, train_class2, test_class1, test_class2){
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
      
      res = auto.fit.legen_fast(X_li_eval[[num_obs]], c_1, b_1, m = 500, method = "LOOCV", threshold = 0, inte = FALSE)
      b_x[num_obs] = res$BC[[2]]
      coeffi_list[[num_obs]] = res$Estimate
      
    } else{
      res = auto.fit.legen_fast(X_li_eval[[num_obs]], c_2, b_2, m = 500, method = "LOOCV", threshold = 0, inte = FALSE) 
      b_y[(num_obs-train_class1)] = res$BC[[2]]
      coeffi_list[[num_obs]] = res$Estimate
      
    }
  }
  
  for(num_obs in 1:total_num_train){
    if(num_obs <= train_class1){
      b_end = floor(b_x[num_obs]/2)
      b_start = 1
      
      D_j_list = unlist(lapply(coeffi_list[[num_obs]], function(x) max(x) - min(x))) 
      ma_diff[num_obs]  = max(D_j_list[b_start:b_end])
    } else{
      b_end = floor(b_y[num_obs-train_class1]/2)
      b_start = 1
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
    res = auto.fit.legen_fast(X_li_test[[num_obs]], c_1, b_1, m = 500, method = "LOOCV", threshold = 0, inte = FALSE)
    b_z[num_obs] = res$BC[[2]]
    coeffi_list[[num_obs]] = res$Estimate
  }
  
  
  for(num_obs in 1:total_num_test){
    b_end = floor(b_z[num_obs]/2)
    b_start = 1
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

# Model 1,2,a,b,c
model_name = c(1,2,"a","b","c")
for(case in 1:5){
  print(paste("Model", model_name[case]))
  res = c()
  for(j in 1:500){
    res[j] = sim_func_auto_faster_featurehalf(1600, case, 1:5, 1:4, 1:5, 1:4, 100, 100, 25, 25)
  }
  print(round(mean(res),2))
  print(round(sd(res), 2))
  
}

# Real Data 
sim_func_auto_faster_featurehalf_realdata = function(Train_list, Testing_list){
  # using auto.select 
  test_class1 = length(Testing_list)/2
  test_class2 = length(Testing_list)/2 
  train_class1 = length(Train_list)/2
  train_class2 = length(Train_list)/2
  
  total_num_test = test_class1 + test_class2
  total_num_train = train_class1 + train_class2
  
  ma_diff = c()
  para_c_b = list()
  
  b_x = c()
  b_y = c()
  coeffi_list = list()
  
  for(num_obs in 1:total_num_train){
    if(num_obs <= train_class1){
      
      res = auto.fit.legen_fast(Train_list[[num_obs]], 1:10, 1:4, m = 500, method = "LOOCV")
      b_x[num_obs] = res$BC[[2]]
      coeffi_list[[num_obs]] = res$Estimate
      
    } else{
      res = auto.fit.legen_fast(Train_list[[num_obs]], 1:10, 1:4, m = 500, method = "LOOCV") 
      b_y[(num_obs-train_class1)] = res$BC[[2]]
      coeffi_list[[num_obs]] = res$Estimate
      
    }
  }
  
  b_star_x = min(b_x)
  b_star_y = min(b_y)
  
  
  
  for(num_obs in 1:total_num_train){
    if(num_obs <= train_class1){
      b_end = floor(b_x[num_obs]/2)
      b_start = 1
      
      D_j_list = unlist(lapply(coeffi_list[[num_obs]], function(x) max(x) - min(x)))
      D_j_list = D_j_list[2:length(D_j_list)]
      ma_diff[num_obs]  = max(D_j_list[b_start:b_end])
    } else{
      b_end = floor(b_y[num_obs-train_class1]/2)
      b_start = 1
      D_j_list = unlist(lapply(coeffi_list[[num_obs]], function(x) max(x) - min(x))) 
      D_j_list = D_j_list[2:length(D_j_list)]
      ma_diff[num_obs]  = max(D_j_list[b_start:b_end])
    }
  }
  
  med_class1 = median(ma_diff[1:train_class1])
  med_class2 = median(ma_diff[(1+train_class1):total_num_train])
  Y = c(rep(1, train_class1), rep(2, train_class2))
  
  diff = seq(min(ma_diff)/2, 2*max(ma_diff), length.out = 10000)
  result = c()
  
  for(j in 1:10000){
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
  
  threshold = round(diff[which.max(result)], 2) 
  
  if(length(threshold) > 1){
    threshold = threshold[1]
  }
  
  # test set 
  
  
  ma_diff_test = c()
  b_z = c()
  coeffi_list = list()
  
  for(num_obs in 1:total_num_test){
    res = auto.fit.legen_fast(Testing_list[[num_obs]], 1:10, 1:3, m = 500, method = "LOOCV")
    b_z[num_obs] = res$BC[[2]]
    coeffi_list[[num_obs]] = res$Estimate
  }
  
  b_star_z = min(b_z)
  
  
  
  for(num_obs in 1:total_num_test){
    b_end = floor(b_z[num_obs]/2)
    b_start = 1
    D_j_list = unlist(lapply(coeffi_list[[num_obs]], function(x) max(x) - min(x)))
    D_j_list = D_j_list[2:length(D_j_list)]
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


# Real Data 

# CZ train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_CZ.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_CZ.rds")

# classification 
print("Channel CZ accuracy:")
print(sim_func_auto_faster_featurehalf_realdata(X_li_train, X_li_test))


