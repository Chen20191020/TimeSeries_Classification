
generate_AR1_class1 = function(n, v){
  ts = c()
  x_ini = rnorm(1, 0, 1/v)
  for(i in 1:n){
    if(i == 1){
      ts[i] = 0.4*cos(2*pi*(i/n))*x_ini  + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] = 0.4*cos(2*pi*(i/n))*ts[i-1] + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}

generate_AR1_class2 = function(n, v){
  ts = c()
  x_ini = rnorm(1, 0, 1/v)
  for(i in 1:n){
    if(i == 1){
      ts[i] = 0.2*cos(2*pi*(i/n))*x_ini + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] = 0.2*cos(2*pi*(i/n))*ts[i-1] + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}


function_names <- c(
  "generate_AR1_class1", "generate_AR1_class2"
)

n_vary = c(500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500)

# The accuracy for n values from 500 to 1500, in order.
for(case in 1:11){
  res = c()
  n = n_vary[case]
  for(j in 1:500){
    res[j] = sim_func_without_auto_faster(n, 1, 4, 1, 4, 1, 100, 100, 25, 25)
  }
  
  print(round(mean(res),2))
  print(round(sd(res), 2))
  
}
