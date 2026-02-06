generate_AR1_class1 = function(n, v){
  ts = c()
  x_ini = rnorm(1, 0, 1/v)
  for(i in 1:n){
    if(i == 1){
      ts[i] = 0.4*cos(2*pi*(i/n))*x_ini  + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] = 0.4*cos(2*pi*(i/n))*ts[i-1] + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}

generate_AR1_class2 = function(n, v){
  ts = c()
  x_ini = rnorm(1, 0, 1/v)
  for(i in 1:n){
    if(i == 1){
      ts[i] = 0.2*cos(2*pi*(i/n))*x_ini + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] = 0.2*cos(2*pi*(i/n))*ts[i-1] + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}

# using auto.select 
train_class1 = 25
train_class2 = 25
n = 1024
total_num_train = train_class1 + train_class2
X_li_eval = list()
ma_diff = c()
para_c_b = list()

for(i in 1:total_num_train){
  if(i <= train_class1){
    X_li_eval[[i]] = generate_AR1_class1(n, 1)
  } else{
    X_li_eval[[i]] = generate_AR1_class2(n, 1)
  }
}

coeffi_list = list()

for(num_obs in 1:total_num_train){
  if(num_obs <= train_class1){
    res =  fix.fit.legen_fast(X_li_eval[[num_obs]], 4, 1, 500, inte = FALSE)
    coeffi_list[[num_obs]] = res$ts.coef
    
  } else{
    res = fix.fit.legen_fast(X_li_eval[[num_obs]], 4, 1, 500, inte = FALSE) 
    coeffi_list[[num_obs]] = res$ts.coef
  }
  
}

b_star_x = 1
b_star_y = 1

for(num_obs in 1:total_num_train){
  if(num_obs <= train_class1){
    b_end = 1
    b_start = max(b_end - b_star_x + 1, b_star_x)
    
    D_j_list = unlist(lapply(coeffi_list[[num_obs]], function(x) max(x) - min(x))) 
    ma_diff[num_obs]  = max(D_j_list[b_start:b_end])
  } else{
    b_end = 1
    b_start = max(b_end - b_star_y + 1, b_star_y)
    D_j_list = unlist(lapply(coeffi_list[[num_obs]], function(x) max(x) - min(x))) 
    ma_diff[num_obs]  = max(D_j_list[b_start:b_end])
  }
}

feature = ma_diff
print(feature)

