
# Model 17

f = function(t, a, t1, t2){
  # We subtract t1 inside the sine function
  ifelse(t >= t1 & t <= t2, 0.4 * cos(2 * pi * (t - t1)), a)
}
generate_illAR1_class1 = function(n, v, a, t1, t2){
  ts = c()
  x_ini = rnorm(1, 0, 1/v)
  for(i in 1:n){
    if(i == 1){
      ts[i] = f(i/n, a, t1, t2)*x_ini + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] = f(i/n, a, t1, t2)*ts[i-1] + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}

generate_illAR1_class2 = function(n, v, a, t1, t2){
  ts = c()
  x_ini = rnorm(1, 0, 1/v)
  for(i in 1:n){
    if(i == 1){
      ts[i] = f(i/n, a, t1, t2)*x_ini + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] = f(i/n, a, t1, t2)*ts[i-1] + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}


a = 0
train_class1 = 100
train_class2 = 100
test_class1 = 25
test_class2 = 25
n = 1024
c_1 = 4
b_1 = 1

c_2 = 4
b_2 = 1
# using auto.select 
total_num_test = test_class1 + test_class2
total_num_train = train_class1 + train_class2

t_11 = 0
t_22 = 1

t_12 = c(0.1, 0.2, 0.3, 0.4)
t_21 = c(0.9, 0.8, 0.7, 0.6)

for(model_case in 1:4){
  accuracy_rate = c()
  # print the change point for each case [c_x, d_x], [c_y, d_y]
  print(c(0,t_12[model_case],t_21[model_case],1))
  for(num_i in 1:500){
    
    X_li_eval = list()
    ma_diff = c()
    para_c_b = list()
    
    for(i in 1:total_num_train){
      if(i <= train_class1){
        X_li_eval[[i]] = generate_illAR1_class1(n, 1, a, t_11, t_12[model_case])
      } else{
        X_li_eval[[i]] = generate_illAR1_class2(n, 1, a, t_21[model_case], t_22)
      }
    }
    
    for(num_obs in 1:total_num_train){
      if(num_obs <= train_class1){
        res = fix.fit.legen_fast(X_li_eval[[num_obs]], c_1, b_1, 500) #
        ma_diff[num_obs] = max(res$ts.coef[[b_1]]) - min(res$ts.coef[[b_1]])
      } else{
        res = fix.fit.legen_fast(X_li_eval[[num_obs]], c_2, b_2, 500)
        ma_diff[num_obs] = max(res$ts.coef[[b_2]]) - min(res$ts.coef[[b_2]])
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
        X_li_test[[i]] = generate_illAR1_class1(n, 1, a, t_11, t_12[model_case])
      } else{
        X_li_test[[i]] = generate_illAR1_class2(n, 1, a, t_21[model_case], t_22)
      }
    }
    
    ma_diff_test = c()
    
    for(num_obs in 1:total_num_test){
      
      if(num_obs <= test_class1){
        res = fix.fit.legen_fast(X_li_test[[num_obs]], c_1, b_1, 500)
        ma_diff_test[num_obs] = max(res$ts.coef[[b_1]]) - min(res$ts.coef[[b_1]])
      } else{
        res = fix.fit.legen_fast(X_li_test[[num_obs]], c_2, b_2, 500)
        ma_diff_test[num_obs] = max(res$ts.coef[[b_2]]) - min(res$ts.coef[[b_2]])
      }
      
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
    accuracy_rate[num_i] = result/(test_class1 + test_class2)
  }
  print(c(mean(accuracy_rate), sd(accuracy_rate)))
}
