#  Simulation

# Epsilon setting (a)


for(case in 1:3){
  res = c()
  print(paste("Model", case+6))
  c = c(5,4, 5,4, 5,5)
  b = c(2,2, 3,3, 1,1)
  
  for(j in 1:500){
    res[j] = sim_func_without_auto_faster(1024, case, c[2*case-1], b[2*case-1], c[2*case], b[2*case], 50, 250, 25, 25)
  }
  
  print(round(mean(res),2))
  print(round(sd(res), 2))
  
}


