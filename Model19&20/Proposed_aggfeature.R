

# Model unit root and long memory

model_name = c("unit root","long memory")

for(case in 1:2){
  res = c()
  print(paste("Model", model_name[case]))
  for(j in 1:500){
    res[j] = sim_func_auto_faster(1600, case, 1:5, 1:4, 1:5, 1:4, 100, 100, 25, 25)
  }
  print(round(mean(res),2))
  print(round(sd(res), 2))
  
}
