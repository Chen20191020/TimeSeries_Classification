#  Simulation
sim_func_without_auto_mean_diff = function(n, c_setting, type, c_1, b_1, c_2, b_2, train_class1, train_class2, test_class1, test_class2, beta = 0.35){
  total_num_test = test_class1 + test_class2
  total_num_train = train_class1 + train_class2
  X_li_eval = list()
  a_mean_func_1 = list()
  a_mean_func_2 = list()
  
  
  for(i in 1:total_num_train){
    if(i <= train_class1){
      X_li_eval[[i]] = get(function_names[2*c_setting-1])(n, 1, beta = beta)
    } else{
      X_li_eval[[i]] = get(function_names[2*c_setting])(n, 1, beta = beta)
    }
  }
  
  
  for(num_obs in 1:total_num_train){
    aux_index_class1 = 1 
    aux_index_class2 = 1
    if(num_obs <= train_class1){
      res = fix.fit.legen_fast(X_li_eval[[num_obs]], c_1, b_1, 500, inte = TRUE)
      a_mean_func_1[[aux_index_class1]] = res$ts.coef[[1]]
      aux_index_class1 = aux_index_class1 + 1
    } else{
      res = fix.fit.legen_fast(X_li_eval[[num_obs]], c_2, b_2, 500, inte = TRUE)
      a_mean_func_2[[aux_index_class2]]= res$ts.coef[[1]] 
      aux_index_class2 = aux_index_class2 + 1
    }
    
  }
  
  mean_func_1 = Reduce(`+`, a_mean_func_1)/length(a_mean_func_1)
  mean_func_2 = Reduce(`+`, a_mean_func_2)/length(a_mean_func_2)
  
  Y = c(rep(1, train_class1), rep(2, train_class2))
  
  
  
  # test set 
  
  X_li_test = list()
  
  for(i in 1:total_num_test){
    if(i <= total_num_test/2){
      X_li_test[[i]] = get(function_names[2*c_setting-1])(n, 1, beta = beta)
    } else{
      X_li_test[[i]] = get(function_names[2*c_setting])(n, 1, beta = beta)
    }
  }
  
  mean_func_test = list()
  
  for(num_obs in 1:total_num_test){
    
    if(num_obs <= test_class1){
      
      res = fix.fit.legen_fast(X_li_test[[num_obs]], c_1, b_1, 500, inte = TRUE)
      mean_func_test[[num_obs]] = res$ts.coef[[1]]
    } else{
      res = fix.fit.legen_fast(X_li_test[[num_obs]], c_2, b_2, 500, inte = TRUE)
      mean_func_test[[num_obs]] = res$ts.coef[[1]] 
    }
    
  }
  
  Y = c(rep(1, test_class1), rep(2, test_class2))
  result = c()
  y_pre = rep(0, length(mean_func_test))
  
  for(num_obs in 1:length(mean_func_test)){
    med_class1 = max(abs(mean_func_test[[num_obs]] - mean_func_1))
    med_class2 = max(abs(mean_func_test[[num_obs]] - mean_func_2))
    if(med_class1 <= med_class2){
      y_pre[num_obs] = 1
    } else{
      y_pre[num_obs] = 2
    }
    
  }
  result = sum(y_pre == Y)
  return(result/total_num_test)
}

# Epsilon setting (a)


for(case in 1:6){
  res = c()
  print(paste("Model", as.roman(case)))
  c = c(4,4, 4,4, 3,4, 4,4, 2,4, 3,4)
  b = c(1,1, 2,2, 2,2, 1,1, 1,1, 1,1)
  
  for(j in 1:500){
    res[j] = sim_func_without_auto_mean_diff(1024, case, "Legen", c[2*case-1], b[2*case-1], c[2*case], b[2*case], 50, 250, 25, 25)

  }
  
  print(round(mean(res),2))
  print(round(sd(res), 2))
  
}


