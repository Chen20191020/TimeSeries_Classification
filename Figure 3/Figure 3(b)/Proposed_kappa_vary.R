# Algo for simulation  

sim_func_without_auto_faster_3a = function(n, c_1, b_1, c_2, b_2, train_class1, train_class2, test_class1, test_class2, delta){
  # using auto.select 
  total_num_test = test_class1 + test_class2
  total_num_train = train_class1 + train_class2
  X_li_eval = list()
  ma_diff = c()
  para_c_b = list()
  
  for(i in 1:total_num_train){
    if(i <= train_class1){
      X_li_eval[[i]] = generate_AR1_test_class1(n, 1, delta)
    } else{
      X_li_eval[[i]] = generate_AR1_test_class2(n, 1, delta)
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
      X_li_test[[i]] = generate_AR1_test_class1(n, 1, delta)
    } else{
      X_li_test[[i]] = generate_AR1_test_class2(n, 1, delta)
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


generate_AR1_test_class1 = function(n, v, delta){
  ts = c()
  x_ini = rnorm(1,0,1)
  for(i in 1:n){
    if(i == 1){
      ts[i] = 2*delta*cos(2*pi*(i/n))*x_ini  + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] = 2*delta*cos(2*pi*(i/n))*ts[i-1] + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}

generate_AR1_test_class2 = function(n, v, delta){
  ts = c()
  x_ini = rnorm(1,0,1)
  for(i in 1:n){
    if(i == 1){
      ts[i] = delta*cos(2*pi*(i/n))*x_ini + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] = delta*cos(2*pi*(i/n))*ts[i-1] + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}



sum_res_proposed = c()
sum_res_delta = list()

n_del = seq(1, 5, by = 0.5)
del = 0.2

for(ind_del in 1:length(n_del)){
  
  sum_res_proposed = c()
  
  for(simu_num in 1:500){
    sum_res_proposed[simu_num] = sim_func_without_auto_faster_3a(1024, 4, 1, 4, 1, 50, 50*n_del[ind_del], 25, 25, delta = del)
  }
  
  sum_res_delta[[ind_del]] = sum_res_proposed
  print(ind_del)
}

# The accuracy for kappa values from 1 to 5, in order.
print(round(unlist(lapply(sum_res_delta, mean)), 3))
print(round(unlist(lapply(sum_res_delta, sd)), 3))
