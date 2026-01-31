#  Simulation

# Epsilon setting (a)
# case 1

function_names <- c(
  "generate_AR1_class1", "generate_AR1_class2",
  "generate_AR2_class1", "generate_AR2_class2",
  "generate_nAR1_class1", "generate_MA2_class2",
  "generate_nAR1.2_class1", "generate_nAR1.2_class2",
  "generate_AR2.2_class1", "generate_AR2.2_class2",
  "generate_nAR2_class1", "generate_nAR2_class2"
)

#eps3: model6


for(case in 1:6){
  res = c()
  c = c(5,5, 5,5, 4,2, 5,4, 2,4, 4,5)
  b = c(1,1, 2,2, 1,2, 2,2, 2,2, 2,2)
  
  for(j in 1:500){
    res[j] = sim_func_without_auto_faster(1024, case, c[2*case-1], b[2*case-1], c[2*case], b[2*case], 50, 250, 25, 25)
    if(j %in% c(1, 200, 500)){
      print(res[j])
    }
  }
  
  print(round(mean(res),2))
  print(round(sd(res), 2))
  
}


