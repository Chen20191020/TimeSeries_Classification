

# Balanced dataset 





function_names <- c(
  "generate_AR1_class1", "generate_AR1_class2",
  "generate_AR2_class1", "generate_AR2_class2",
  "generate_nAR1_class1", "generate_MA2_class2",
  "generate_nAR1.2_class1", "generate_nAR1.2_class2",
  "generate_AR2.2_class1", "generate_AR2.2_class2",
  "generate_nAR2_class1", "generate_nAR2_class2"
)



case_res = list()
sum_res = c()
N_x_train = 250
N_y_train = 250
N_x_test = 25 
N_y_test = 25
N_total_train = N_x_train + N_y_train
N_total_test = N_x_test + N_y_test

Y = as.factor(c(rep("abnormal", N_x_train), rep("normal", N_y_train)))
Y_test = as.factor(c(rep("abnormal", N_x_test), rep("normal", N_y_test)))

for(case_num in 1:6){
  for(simu_num in 1:500){
    X_li_eval = list()
    for(i in 1:N_total_train){
      if(i <= N_x_train){
        X_li_eval[[i]] = get(function_names[2*case_num - 1])(2000, 1)
      } else{
        X_li_eval[[i]] = get(function_names[2*case_num])(2000, 1)
      }
    }
    
    # Convert the list to a data frame
    df_train <- do.call(rbind, lapply(X_li_eval, function(x) data.frame(t(x))))
    
    # RF
    classifier_RF = randomForest(x = df_train, y = Y, ntree = 500) 
    
    # testing 
    X_li_test = list()
    
    for(i in 1:N_total_test){
      if(i <= N_x_test){
        X_li_test[[i]] = get(function_names[2*case_num - 1])(2000, 1)
      } else{
        X_li_test[[i]] = get(function_names[2*case_num])(2000, 1)
      }
    }
    
    df_test <- do.call(rbind, lapply(X_li_test, function(x) data.frame(t(x))))
    
    
    # Predicting the Test set results 
    y_pred = predict(classifier_RF, newdata = df_test) 
    
    # Confusion Matrix 
    confusion_mtx = table(Y_test, y_pred) 
    print(simu_num)
    sum_res[simu_num] = (confusion_mtx[1,1] + confusion_mtx[2,2])/total_num_test
  }
  case_res[[case_num]] = sum_res
}

lapply(case_res, mean)
lapply(case_res, sd)



