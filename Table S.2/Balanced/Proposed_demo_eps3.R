#  Simulation

# Epsilon setting (c)


# L2 methods
sim_func_auto_faster_L2Dis = function(Train_list, Testing_list, c = 25, b = 1, m = 500) {
  test_class1 = length(Testing_list) / 2
  test_class2 = length(Testing_list) / 2
  train_class1 = length(Train_list) / 2
  train_class2 = length(Train_list) / 2
  
  total_num_test = test_class1 + test_class2
  total_num_train = train_class1 + train_class2
  
  coeffi_list = list()
  
  for (num_obs in 1:total_num_train) {
    res = fix.fit.legen_fast(Train_list[[num_obs]], c, b, m, inte = FALSE)
    coeffi_list[[num_obs]] = res$ts.coef
  }
  
  coeffi_list_class1 = coeffi_list[1:train_class1]
  coeffi_list_class2 = coeffi_list[(1 + train_class1):total_num_train]
  
  num_lags = b
  
  coeff_avg_class1 = lapply(1:num_lags, function(k) {
    lag_k_list = lapply(coeffi_list_class1, function(obs) obs[[k]])
    Reduce("+", lag_k_list) / train_class1
  })
  
  coeff_avg_class2 = lapply(1:num_lags, function(k) {
    lag_k_list = lapply(coeffi_list_class2, function(obs) obs[[k]])
    Reduce("+", lag_k_list) / train_class2
  })
  
  coeffi_list_test = list()
  for (num_obs in 1:total_num_test) {
    res = fix.fit.legen_fast(Testing_list[[num_obs]], c, b, m, inte = FALSE)
    coeffi_list_test[[num_obs]] = res$ts.coef
  }
  
  Y = c(rep(1, test_class1), rep(2, test_class2))
  y_pre = rep(0, total_num_test)
  
  for (num_obs in 1:total_num_test) {
    L2_i_class1 = sum(sapply(1:num_lags, function(k) {
      sum((coeffi_list_test[[num_obs]][[k]] - coeff_avg_class1[[k]])^2/m)
    }))
    
    L2_i_class2 = sum(sapply(1:num_lags, function(k) {
      sum((coeffi_list_test[[num_obs]][[k]] - coeff_avg_class2[[k]])^2/m)
    }))
    
    y_pre[num_obs] = ifelse(L2_i_class1 < L2_i_class2, 1, 2)
  }
  
  result = sum(y_pre == Y)
  return(list(coeff_avg_class1, coeff_avg_class2, result / total_num_test))
}


# Model 1-7 concentration index 

# the threshold 2*log(500)/500

#train_class_size = 100
#test_class_size = 25
#n = 1024
#model_index = 1:7
#b_choose = c(1,2,2,2,2,2,1)

# for(m_index in model_index){
#   res = c()
#   print(paste("Model", m_index))
#   for(ex_t in 1:1){
#     Train_list = list()
#     for (i in 1:(2 * train_class_size)) {
#       if (i <= train_class_size) {
#         Train_list[[i]] = get(function_names[2*m_index-1])(n, 1)
#       } else {
#         Train_list[[i]] = get(function_names[2*m_index])(n, 1)
#       }
#     }
#     
#     Testing_list = list()
#     for (i in 1:(2 * test_class_size)) {
#       if (i <= test_class_size) {
#         Testing_list[[i]] = get(function_names[2*m_index-1])(n, 1)
#       } else {
#         Testing_list[[i]] = get(function_names[2*m_index])(n, 1)
#       }
#     }
#     print(paste("Concentration", concentration_index(Train_list, train_class_size, c=25, b=b_choose[m_index])[[5]]))
#   }
# }



# Model 1-7 Accuracy rate

acc_res = c()
sd_res = c()
train_class_size = 100
test_class_size = 25
n = 1024
c_choose = c(4,5,5,5,5,5)
b_choose = c(1,2,2,2,2,2)
model_index = 1:6
for(m_index in model_index){
  res = c()
  print(paste("Model", m_index))
  for(ex_t in 1:500){
    Train_list = list()
    for (i in 1:(2 * train_class_size)) {
      if (i <= train_class_size) {
        Train_list[[i]] = get(function_names[2*m_index-1])(n, 1)
      } else {
        Train_list[[i]] = get(function_names[2*m_index])(n, 1)
      }
    }
    
    Testing_list = list()
    for (i in 1:(2 * test_class_size)) {
      if (i <= test_class_size) {
        Testing_list[[i]] = get(function_names[2*m_index-1])(n, 1)
      } else {
        Testing_list[[i]] = get(function_names[2*m_index])(n, 1)
      }
    }
    res[ex_t] = sim_func_auto_faster_L2Dis(Train_list, Testing_list, c_choose[m_index], b_choose[m_index])[[3]]# 
    
  }
  acc_res[m_index] = mean(res) 
  sd_res[m_index] = sd(res)
  print(c(mean(res), sd(res)))
}

for(case in 7:7){
  print(paste("Model", case))
  res = c()
  for(j in 1:500){
    res[j] = sim_func_without_auto_faster(1024, case, 2, 1, 8, 1, 100, 100, 25, 25)
  }
  print(round(mean(res),2))
  print(round(sd(res), 2))
  
}
