library(wavethresh)

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


# LSW demo

train_class1 = 100
train_class2 = 100
total_num_train = train_class1 + train_class2
n = 1024
J = log(n, 2) - 1 
f_propotion = 0.04


n_del= seq(0.05, 0.4, by = 0.05)


# Print the accuracy for delta values from 0.05 to 0.40, in order.

for(ind_del in 1:length(n_del)){
  acuracy = c()
  for(repea in 1:500){
    X_li_eval = list()
    for(i in 1:total_num_train){
      if(i <= train_class1){
        X_li_eval[[i]] = generate_AR1_test_class1(n, 1, n_del[ind_del])
      } else{
        X_li_eval[[i]] = generate_AR1_test_class2(n, 1, n_del[ind_del])
      }
    }
    
    X_li_s = list()
    for(i in 1:total_num_train){
      X_li_s[[i]] = ewspec(X_li_eval[[i]])
    }
    
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
    
    top_n <- f_propotion*length(all_values)
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
    
    n_test = 50
    y_pre = c()
    for(i in 1:n_test){
      if(i <= (n_test/2)){
        ts_classi = generate_AR1_test_class1(n, 1, n_del[ind_del])
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
        ts_classi = generate_AR1_test_class2(n, 1, n_del[ind_del])
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
    acuracy[repea] = (sum(y_pre[1:(n_test/2)] == 0) + sum(y_pre[(n_test/2+1):n_test] == 1))/n_test

  }
  print(round(c(mean(acuracy), sd(acuracy)),3))
}
