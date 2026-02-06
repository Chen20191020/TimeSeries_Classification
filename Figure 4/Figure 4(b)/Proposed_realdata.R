

#  classfication for EEG 

# C3 train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_C3.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_C3.rds")

# classification  
ma_diff_train = c()
bc_list_train_fast = list()
bc_list_test_fast = list()
for(num_obs in 1:504){
  res = auto.fit.legen_fast(X_li_train[[num_obs]], c(1:10), c(1:4), method = "LOOCV")
  bc_list_train_fast[[num_obs]] = res[[4]]
  res_coef = res[[1]]
  phi_j = res_coef[[length(res_coef)]]
  ma_diff_train[num_obs] = max(phi_j) - min(phi_j)
}
Y = c(rep("abnormal", 504/2), rep("normal", 504/2))
diff = seq(1/2*min(ma_diff_train), 2*max(ma_diff_train), length.out = 10000)
result = c()
for(j in 1:10000){
  y_pre = rep(0, 504)
  for(num_obs in 1:504){
    y_pre[num_obs] = ifelse(ma_diff_train[num_obs] > diff[j], "abnormal", "normal")
  }
  result[j] = sum(y_pre == Y)
}
threshold = diff[which.max(result)]
# test 
ma_diff = c()
for(num_obs in 1:length(X_li_test)){
  res = auto.fit.legen_fast(X_li_test[[num_obs]], c(1:11), c(1:3), method = "LOOCV", inte = T)
  bc_list_test_fast[[num_obs]] = res[[4]]
  res_coef = res[[1]]
  phi_j = res_coef[[length(res_coef)]]
  ma_diff[num_obs] = max(phi_j) - min(phi_j)
}
result = c() 
Y = c(rep("abnormal", length(X_li_test)/2), rep("normal", length(X_li_test)/2))
y_pre = rep(0, length(X_li_test))
for(num_obs in 1:length(X_li_test)){
  y_pre[num_obs] = ifelse(ma_diff[num_obs] > threshold, "abnormal", "normal")
}
result = sum(y_pre == Y)/length(X_li_test)
y_hat_c3 = y_pre

print(paste("Channel C3 accuracy:", round(result, 3)))




# C4 train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_C4.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_C4.rds")

# classification  
ma_diff_train = c()
bc_list_train_fast = list()
bc_list_test_fast = list()
for(num_obs in 1:504){
  res = auto.fit.legen_fast(X_li_train[[num_obs]], c(1:10), c(1:4), method = "LOOCV")
  bc_list_train_fast[[num_obs]] = res[[4]]
  res_coef = res[[1]]
  phi_j = res_coef[[length(res_coef)]]
  ma_diff_train[num_obs] = max(phi_j) - min(phi_j)
}
Y = c(rep("abnormal", 504/2), rep("normal", 504/2))
diff = seq(1/2*min(ma_diff_train), 2*max(ma_diff_train), length.out = 10000)
result = c()
for(j in 1:10000){
  y_pre = rep(0, 504)
  for(num_obs in 1:504){
    y_pre[num_obs] = ifelse(ma_diff_train[num_obs] > diff[j], "abnormal", "normal")
  }
  result[j] = sum(y_pre == Y)
}
threshold = diff[which.max(result)]
# test 
ma_diff = c()
for(num_obs in 1:length(X_li_test)){
  res = auto.fit.legen_fast(X_li_test[[num_obs]], c(1:11), c(1:3), method = "LOOCV")
  bc_list_test_fast[[num_obs]] = res[[4]]
  res_coef = res[[1]]
  phi_j = res_coef[[length(res_coef)]]
  ma_diff[num_obs] = max(phi_j) - min(phi_j)
}
result = c() 
Y = c(rep("abnormal", length(X_li_test)/2), rep("normal", length(X_li_test)/2))
y_pre = rep(0, length(X_li_test))
for(num_obs in 1:length(X_li_test)){
  y_pre[num_obs] = ifelse(ma_diff[num_obs] > threshold, "abnormal", "normal")
}
result = sum(y_pre == Y)/length(X_li_test)
y_hat_c4 = y_pre
print(paste("Channel C4 accuracy:", round(result, 3)))




# CZ train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_CZ.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_CZ.rds")

# classification  
ma_diff_train = c()
bc_list_train_fast = list()
bc_list_test_fast = list()
for(num_obs in 1:504){
  res = auto.fit.legen_fast(X_li_train[[num_obs]], c(1:10), c(1:4), method = "LOOCV")
  bc_list_train_fast[[num_obs]] = res[[4]]
  res_coef = res[[1]]
  phi_j = res_coef[[length(res_coef)]]
  ma_diff_train[num_obs] = max(phi_j) - min(phi_j)
}
Y = c(rep("abnormal", 504/2), rep("normal", 504/2))
diff = seq(1/2*min(ma_diff_train), 2*max(ma_diff_train), length.out = 10000)
result = c()
for(j in 1:10000){
  y_pre = rep(0, 504)
  for(num_obs in 1:504){
    y_pre[num_obs] = ifelse(ma_diff_train[num_obs] > diff[j], "abnormal", "normal")
  }
  result[j] = sum(y_pre == Y)
}
threshold = round(diff[which.max(result)], 2)
# test 
ma_diff = c()
for(num_obs in 1:length(X_li_test)){
  res = auto.fit.legen_fast(X_li_test[[num_obs]], c(1:10), c(1:3), method = "LOOCV", inte = T)
  bc_list_test_fast[[num_obs]] = res[[4]]
  res_coef = res[[1]]
  phi_j = res_coef[[length(res_coef)]]
  ma_diff[num_obs] = max(phi_j) - min(phi_j)
}
result = c() 
Y = c(rep("abnormal", length(X_li_test)/2), rep("normal", length(X_li_test)/2))
y_pre = rep(0, length(X_li_test))
for(num_obs in 1:length(X_li_test)){
  y_pre[num_obs] = ifelse(ma_diff[num_obs] > threshold, "abnormal", "normal")
}
result = sum(y_pre == Y)/length(X_li_test)
y_hat_cz = y_pre

print(paste("Channel CZ accuracy:", round(result, 3)))

# Majority voting 
# 1. Combine vectors into a matrix
matrix_preds <- cbind(as.character(y_hat_cz), 
                      as.character(y_hat_c3), 
                      as.character(y_hat_c4))
# 2. Compute Majority
# Count how many classifiers predicted "abnormal" for each row
votes_abnormal <- rowSums(matrix_preds == "abnormal")
# If "abnormal" has 2 or more votes, the majority is "abnormal". Otherwise, "normal".
majority <- ifelse(votes_abnormal >= 2, "abnormal", "normal")

print(paste("Majority voting accuracy:", round(sum(majority == Y)/150, 3)))



