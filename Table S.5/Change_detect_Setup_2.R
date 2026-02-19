
#  Model 17

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

detect_change_points = function(data, c_candi = 1:2, d_candi = 1, m = 1000, delta = 0.0001, c_point_candi = seq(0.1, 0.9, 0.1), grid = seq(0, 1, 0.1)){
  n_changepoint = length(data)
  epsilon = 0.1
  
  diff_score = c()
  
  #c_point_candi = seq(0.2, 0.8, 0.2)
  #grid = seq(0, 1, 0.2)
  
  #c_point_candi = seq(0.3, 0.7, 0.2)
  #grid = seq(0.1, 0.9, 0.2)
  
  #c_point_candi = seq(0.1, 0.9, 0.1)
  #grid = seq(0, 1, 0.1)
  aux_ind = 1
  
  for(i in 1:(length(grid)-2)){
    inte_1 = grid[i]
    inte_2 = grid[i+1] 
    inte_3 = grid[i+2]
    
    data_left = ts[(inte_1*n_changepoint+1):(inte_2*n_changepoint)]
    res = auto.fit.legen_fast(data_left, c_candi, d_candi, m = m, method = "LOOCV", inte = FALSE)
    f_esti_left = res$Estimate[[1]]
    x_i_first = seq(inte_1, inte_2, length.out = m)
    h_minus = (f_esti_left[which.min(abs(x_i_first - (inte_2-delta)))] - f_esti_left[which.min(abs(x_i_first - (inte_2)))])/-delta
    
    
    data_right = ts[(inte_2*n_changepoint+1):(inte_3*n_changepoint)]
    res = auto.fit.legen_fast(data_right, c_candi, d_candi, m = m, method = "LOOCV", inte = FALSE)
    f_esti_right = res$Estimate[[1]]
    x_i_second = seq(inte_2, inte_3, length.out = m)
    
    h_plus = (f_esti_right[which.min(abs(x_i_second - (inte_2+delta)))] - f_esti_left[which.min(abs(x_i_first - (inte_2)))])/delta
    
    diff_score[aux_ind] = abs(h_minus - h_plus)
    aux_ind = aux_ind + 1
  }
  
  change_point = diff_score[which.max(diff_score)]
  if(change_point > epsilon){
    return(c_point_candi[which.max(diff_score)])
    
  } else{
    return(-1)
  }
}

t_12 = c(0.1, 0.2, 0.3, 0.4)
t_21 = c(0.9, 0.8, 0.7, 0.6)

c_m = c(2,2,3,2)


# Setting 1 ([0, 0.1], [0.9, 1]) and Setting 4 ( [0, 0.4], [0.6, 1] )
for(model_case in c(1,4)){
  # print the change point for each case [c_x, d_x], [c_y, d_y]
  print(c(0,t_12[model_case],t_21[model_case],1))
  acc_rate = c()
  for(j in 1:500){
    res = c()
    for(i in 1:50){
      if(i <= 25){
        ts = generate_illAR1_class1(15000, 1, 0, 0, t_12[model_case])
        res[i] = detect_change_points(ts, 1:c_m[model_case], 1)
      } else{
        ts = generate_illAR1_class2(15000, 1, 0, t_21[model_case], 1)
        res[i] = detect_change_points(ts, 1:c_m[model_case], 1)
      }
    }
    acc_rate[j] = (sum(abs(res[1:25] - t_12[model_case]) < 2.220446e-16) + sum(abs(res[26:50] - t_21[model_case]) < 2.220446e-16))/50
  }
  
  print(c(mean(acc_rate), sd(acc_rate)))
  
}



# Setting 2 (setting [0, 0.2], [0.8, 1])
print(c(0,0.2, 0.8,1))
acc_rate = c() 
for(j in 1:500){
  res = c()
  for(i in 1:50){
    if(i <= 25){
      ts = generate_illAR1_class1(15000, 1, 0, 0, t_12[2])
      res[i] = detect_change_points(ts, 1:c_m[2], 1, delta = 0.0001, c_point_candi = seq(0.2, 0.8, 0.2), grid = seq(0, 1, 0.2))
    } else{
      ts = generate_illAR1_class2(15000, 1, 0, t_21[2], 1)
      res[i] = detect_change_points(ts, 1:c_m[2], 1, delta = 0.0001, c_point_candi = seq(0.2, 0.8, 0.2), grid = seq(0, 1, 0.2))
    }
  }
  acc_rate[j] = (sum(abs(res[1:25] - t_12[2]) < 2.220446e-16) + sum(abs(res[26:50] - t_21[2]) < 2.220446e-16))/50
}

print(c(mean(acc_rate), sd(acc_rate)))


# Setting 3 (setting [0, 0.3], [0.7, 1])
print(c(0,0.3, 0.7,1))
acc_rate = c()
for(j in 1:500){
  res = c()
  for(i in 1:50){
    if(i <= 25){
      ts = generate_illAR1_class1(15000, 1, 0, 0, t_12[3])
      res[i] = detect_change_points(ts, 1:c_m[3], 1, delta = 0.001, c_point_candi = seq(0.3, 0.7, 0.2), grid = seq(0.1, 0.9, 0.2))
    } else{
      ts = generate_illAR1_class2(15000, 1, 0, t_21[3], 1)
      res[i] = detect_change_points(ts, 1:c_m[3], 1, delta = 0.001, c_point_candi = seq(0.3, 0.7, 0.2), grid = seq(0.1, 0.9, 0.2))
    }
  }
  acc_rate[j] = (sum(abs(res[1:25] - t_12[3]) < 2.220446e-16) + sum(abs(res[26:50] - t_21[3]) < 2.220446e-16))/50
}

print(c(mean(acc_rate), sd(acc_rate)))

