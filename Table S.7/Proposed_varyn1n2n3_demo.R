#  Simulation
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


# Run Sim

N = c(5, 15, 35, 55, 75, 95, 115, 135)
N3 = c(5, 10, 30, 50, 70, 90, 110, 130)

for(i in 1:8){
  for(k in 1:8){
    res = c()
    for(j in 1:500){
      res[j] = sim_func_without_auto_faster(1024, 1, 4, 1, 4, 1, N[k], N[k], N3[i], N3[i])
    }
    print(c(N[k], N[k], N3[i], N3[i]))
    print(c(round(mean(res),2), round(sd(res), 2)))
  }
}

  
 




