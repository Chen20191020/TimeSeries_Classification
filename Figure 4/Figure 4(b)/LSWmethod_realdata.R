library(wavethresh)
#  classfication for EEG 

u_dsample <- function(data, target_size = 4096) {
  n <- length(data)
  if (n <= target_size) {
    stop("Input data must have more than 4096 points.")
  }
  
  indices <- round(seq(1, n, length.out = target_size))
  return(data[indices])
}

train_class1 = 252
train_class2 = 252
total_num_train = train_class1 + train_class2
n = 4096
J = log(n, 2) - 1 
f_propotion = 0.04

# C3 train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_C3.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_C3.rds")

for(j in 1:504){
  X_li_train[[j]] = u_dsample(X_li_train[[j]])
}

for(j in 1:150){
  X_li_test[[j]] = u_dsample(X_li_test[[j]])
}

X_li_s = list()
for(i in 1:total_num_train){
  X_li_s[[i]] = ewspec(X_li_train[[i]])
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

n_test = 150
y_pre = c()
for(i in 1:n_test){
  if(i <= (n_test/2)){
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

y_hat_c3 = y_pre
# classification  
acc = round((sum(y_pre[1:(n_test/2)] == 0) + sum(y_pre[(n_test/2+1):n_test] == 1))/n_test, 3)
print(paste("Channel C3 accuracy:", acc))


# C4 train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_C4.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_C4.rds")

for(j in 1:504){
  X_li_train[[j]] = u_dsample(X_li_train[[j]])
}

for(j in 1:150){
  X_li_test[[j]] = u_dsample(X_li_test[[j]])
}

X_li_s = list()
for(i in 1:total_num_train){
  X_li_s[[i]] = ewspec(X_li_train[[i]])
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

n_test = 150
y_pre = c()
for(i in 1:n_test){
  if(i <= (n_test/2)){
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

# classification  
y_hat_c4 = y_pre
# classification  
acc = round((sum(y_pre[1:(n_test/2)] == 0) + sum(y_pre[(n_test/2+1):n_test] == 1))/n_test, 3)
print(paste("Channel C4 accuracy accuracy:", acc))


# CZ train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_CZ.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_CZ.rds")

for(j in 1:504){
  X_li_train[[j]] = u_dsample(X_li_train[[j]])
}

for(j in 1:150){
  X_li_test[[j]] = u_dsample(X_li_test[[j]])
}


X_li_s = list()
for(i in 1:total_num_train){
  X_li_s[[i]] = ewspec(X_li_train[[i]])
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

n_test = 150
y_pre = c()
for(i in 1:n_test){
  if(i <= (n_test/2)){
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

y_hat_cz = y_pre
# classification  
acc = round((sum(y_pre[1:(n_test/2)] == 0) + sum(y_pre[(n_test/2+1):n_test] == 1))/n_test, 3)
print(paste("Channel CZ accuracy accuracy:", acc))


# Majority voting 
Y_test = c(rep(0, 75), rep(1, 75))
# Combine the vectors into a matrix
matrix = cbind(as.vector(y_hat_cz), as.vector(y_hat_c3), as.vector(y_hat_c4))
# Compute the majority for each element
majority = rowSums(matrix) >= 2
# Convert logical TRUE/FALSE to numeric 1/0
majority = as.numeric(majority)

print(paste("Majority voting accuracy:", round(sum(majority == Y_test)/150, 3)))
