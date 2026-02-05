#  Simulation for Table 1


# Model I
generate_AR1_class1 = function(n, v, beta = 0.35){
  ts = c()
  x_ini = rnorm(1, 0, 1/v)
  for(i in 1:n){
    if(i == 1){
      ts[i] = 2*beta*(sin(2*pi*(i/n)) + 0.5) + 0.4*cos(2*pi*(i/n))*x_ini  + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] = 2*beta*(sin(2*pi*(i/n)) + 0.5) + 0.4*cos(2*pi*(i/n))*ts[i-1] + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}

generate_AR1_class2 = function(n, v, beta = 0.35){
  ts = c()
  x_ini = rnorm(1, 0, 1/v)
  for(i in 1:n){
    if(i == 1){
      ts[i] = beta*(sin(2*pi*(i/n)) + 0.5) + 0.4*cos(2*pi*(i/n))*x_ini  + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] = beta*(sin(2*pi*(i/n)) + 0.5) + 0.4*cos(2*pi*(i/n))*ts[i-1] + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}


# Model II
generate_AR2_class1 = function(n, v, beta = 0.35){
  ts = c()
  w = rnorm(n+2, 0, 1/v)
  
  for(i in 1:n){
    if(i == 1){
      ts[i] =   2*beta*(sin(2*pi*(i/n)) + 0.5) + (0.6*sin(2*pi*(i/n)))*w[1] + 0.4*w[2] + w[i+2]
    } else if(i == 2){
      ts[i] =   2*beta*(sin(2*pi*(i/n)) + 0.5) + (0.6*sin(2*pi*(i/n)))*w[2]  + 0.4*ts[i-1] + w[i+2]
    } else {
      ts[i] =   2*beta*(sin(2*pi*(i/n)) + 0.5) + (0.6*sin(2*pi*(i/n)))*ts[i-2] + 0.4*ts[i-1] + w[i+2]
    }
  }
  return(ts)
}

generate_AR2_class2 = function(n, v, beta = 0.35){
  ts = c()
  w = rnorm(n+2, 0, 1/v)
  
  for(i in 1:n){
    if(i == 1){
      ts[i] = beta*(sin(2*pi*(i/n)) + 0.5) + (0.6*sin(2*pi*(i/n)))*w[1] + 0.4*w[2] + w[i+2]
    } else if(i == 2){
      ts[i] = beta*(sin(2*pi*(i/n)) + 0.5) + (0.6*sin(2*pi*(i/n)))*w[2]  + 0.4*ts[i-1] + w[i+2]
    } else {
      ts[i] = beta*(sin(2*pi*(i/n)) + 0.5) + (0.6*sin(2*pi*(i/n)))*ts[i-2] + 0.4*ts[i-1] + w[i+2]
    }
  }
  return(ts)
}


# Model III

generate_MA2_class1 = function(n, v, beta = 0.35){
  ts = c()
  epsilon = rnorm(n+2, 0, 1/v)
  
  for(i in 3:(n+2)){
    ts[i-2] = 2*beta*(sin(2*pi*((i-2)/n)) + 0.5) + 0.3*epsilon[i-2] + 0.4*epsilon[i-1]  + epsilon[i] #  
  }
  return(ts)
}


generate_MA2_class2 = function(n, v, beta = 0.35){
  ts = c()
  epsilon = rnorm(n+2, 0, 1/v)
  
  for(i in 3:(n+2)){
    ts[i-2] = beta*(sin(2*pi*((i-2)/n)) + 0.5) + 0.3*epsilon[i-2] + 0.4*epsilon[i-1]  + epsilon[i] #  
  }
  return(ts)
}


# Model IV

generate_nAR1.2_class1 = function(n, v, beta = 0.35){
  ts = c()
  x_ini = rnorm(1, 0, 1/v)
  for(i in 1:n){
    if(i == 1){
      ts[i] =  2*beta*(sin(2*pi*((i-2)/n)) + 0.5) + (0.5*cos(2*pi*(i/n)))*exp(-(i/n)*-x_ini^2)  + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] =  2*beta*(sin(2*pi*((i-2)/n)) + 0.5) + (0.5*cos(2*pi*(i/n)))*exp(-(i/n)*ts[i-1]^2) + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}

generate_nAR1.2_class2 = function(n, v, beta = 0.35){
  ts = c()
  x_ini = rnorm(1, 0, 1/v)
  for(i in 1:n){
    if(i == 1){
      ts[i] =  beta*(sin(2*pi*((i-2)/n)) + 0.5) + (0.5*cos(2*pi*(i/n)))*exp(-(i/n)*-x_ini^2)  + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] =  beta*(sin(2*pi*((i-2)/n)) + 0.5) + (0.5*cos(2*pi*(i/n)))*exp(-(i/n)*ts[i-1]^2) + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}


# Model V

generate_AR2.2_class1 = function(n, v, beta = 0){
  ts = c()
  w = rnorm(n+2, 0, 1/v)
  
  for(i in 1:n){
    if(i == 1){
      ts[i] =  0.5 + (0.2*sin(2*pi*(i/n)))*w[1] + 0.2*w[2] + w[i+2]
    } else if(i == 2){
      ts[i] =  0.5 + (0.2*sin(2*pi*(i/n)))*w[2]  + 0.2*ts[i-1] + w[i+2]
    } else {
      ts[i] =  0.5 + (0.2*sin(2*pi*(i/n)))*ts[i-2] + 0.2*ts[i-1] + w[i+2]
    }
  }
  return(ts)
}

generate_AR2.2_class2 = function(n, v, beta = 0.5){
  ts = c()
  w = rnorm(n+2, 0, 1/v)
  
  for(i in 1:n){
    if(i == 1){
      ts[i] =  (0.5 + beta*exp(-((i/n)-1/2)^2)) + (0.2*sin(2*pi*(i/n)))*w[1] + 0.2*w[2] + w[i+2]
    } else if(i == 2){
      ts[i] =  (0.5 + beta*exp(-((i/n)-1/2)^2)) + (0.2*sin(2*pi*(i/n)))*w[2]  + 0.2*ts[i-1] + w[i+2]
    } else {
      ts[i] =  (0.5 + beta*exp(-((i/n)-1/2)^2)) + (0.2*sin(2*pi*(i/n)))*ts[i-2] + 0.2*ts[i-1] + w[i+2]
    }
  }
  return(ts)
}


# Model VI

generate_nAR2_class1 = function(n, v, beta = 0){
  ts = c()
  w = rnorm(n+2, 0, 1/v)
  for(i in 1:n){
    if(i == 1){
      ts[i] =  0.5 + 0.3*(sin(2*pi*(i/n))+1)*w[1] + 0.2*exp(-(i/n)*w[2]^2) + w[i+2]
    } else if(i == 2){
      ts[i] =   0.5 + 0.3*(sin(2*pi*(i/n))+1)*w[2]  + 0.2*exp(-(i/n)*ts[i-1]^2) + w[i+2]
    } else {
      ts[i] =   0.5 + 0.3*(sin(2*pi*(i/n))+1)*ts[i-2] + 0.2*exp(-(i/n)*ts[i-1]^2) + w[i+2]
    }
  }
  return(ts)
}

generate_nAR2_class2 = function(n, v, beta = 0.5){
  ts = c()
  w = rnorm(n+2, 0, 1/v)
  for(i in 1:n){
    if(i == 1){
      ts[i] =  (0.5 + beta*exp(-((i/n)-1/2)^2)) + 0.3*(sin(2*pi*(i/n))+1)*w[1] + 0.2*exp(-(i/n)*w[2]^2) + w[i+2]
    } else if(i == 2){
      ts[i] =  (0.5 + beta*exp(-((i/n)-1/2)^2)) + 0.3*(sin(2*pi*(i/n))+1)*w[2]  + 0.2*exp(-(i/n)*ts[i-1]^2) + w[i+2]
    } else {
      ts[i] =  (0.5 + beta*exp(-((i/n)-1/2)^2)) + 0.3*(sin(2*pi*(i/n))+1)*ts[i-2] + 0.2*exp(-(i/n)*ts[i-1]^2) + w[i+2]
    }
  }
  return(ts)
}


function_names <- c(
  "generate_AR1_class1", "generate_AR1_class2",
  "generate_AR2_class1", "generate_AR2_class2",
  "generate_MA2_class1", "generate_MA2_class2",
  "generate_nAR1.2_class1", "generate_nAR1.2_class2", 
  "generate_AR2.2_class1", "generate_AR2.2_class2", 
  "generate_nAR2_class1", "generate_nAR2_class2"
)

