#  Simulation

# Epsilon setting (a)

for(case in 1:6){
  res = c()
  c = c(5,5, 5,5, 4,2, 5,4, 2,4, 4,5)
  b = c(1,1, 2,2, 1,2, 2,2, 2,2, 2,2)
  
  for(j in 1:500){
    res[j] = sim_func_without_auto_faster(1024, case, c[2*case-1], b[2*case-1], c[2*case], b[2*case], 100, 100, 25, 25)
    if(j %in% c(1, 200, 500)){
      print(res[j])
    }
  }
  
  print(round(mean(res),2))
  print(round(sd(res), 2))
  
}


