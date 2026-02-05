#  Simulation

# Epsilon setting (c)

for(case in 1:6){
  res = c()
  print(paste("Model", case))
  c = c(5,5, 5,5, 4,2, 5,4, 2,4, 2,5)
  b = c(1,1, 2,2, 1,2, 2,2, 2,2, 3,2)
  
  for(j in 1:500){
    res[j] = sim_func_without_auto_faster(1024, case, c[2*case-1], b[2*case-1], c[2*case], b[2*case], 50, 250, 25, 25)
  }
  
  print(round(mean(res),2))
  print(round(sd(res), 2))
  
}


