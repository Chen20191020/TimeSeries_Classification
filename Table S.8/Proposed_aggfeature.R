
# Model 1,2,a,b,c
model_name = c(1,2,"a","b","c")

for(case in 1:5){
  res = c()
  print(paste("Model", model_name[case]))
  for(j in 1:500){
    res[j] = sim_func_auto_faster(1600, case, 1:5, 1:4, 1:5, 1:4, 100, 100, 25, 25)
  }
  print(round(mean(res),2))
  print(round(sd(res), 2))
  
}

# Real Data 


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
print("Channel CZ accuracy:")
print(round(result, 3))

