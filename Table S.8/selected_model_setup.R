

# Worked model 
# Model 1
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


# Model 2

generate_AR2_class1 = function(n, v){
  ts = c()
  w = rnorm(n+2, 0, 1/v)
  
  for(i in 1:n){
    if(i == 1){
      ts[i] =  (0.6*sin(2*pi*(i/n)))*w[1] + 0.4*w[2] + w[i+2]
    } else if(i == 2){
      ts[i] =  (0.6*sin(2*pi*(i/n)))*w[2]  + 0.4*ts[i-1] + w[i+2]
    } else {
      ts[i] =  (0.6*sin(2*pi*(i/n)))*ts[i-2] + 0.4*ts[i-1] + w[i+2]
    }
  }
  return(ts)
}

generate_AR2_class2 = function(n, v){
  ts = c()
  w = rnorm(n+2, 0, 1/v)
  
  for(i in 1:n){
    if(i == 1){
      ts[i] =  0.4*cos(2*pi*(i/n))*w[1] + 0.6*w[2] + w[i+2]
    } else if(i == 2){
      ts[i] =  0.4*cos(2*pi*(i/n))*w[2]  + 0.6*ts[i-1] + w[i+2]
    } else {
      ts[i] =  0.4*cos(2*pi*(i/n))*ts[i-2] + 0.6*ts[i-1] + w[i+2]
    }
  }
  return(ts)
}



# Model a

generate_AR2s2c1_class1 = function(n, v){
  ts = c()
  w = rnorm(n+2, 0, 1/v)
  
  for(i in 1:n){
    if(i == 1){
      ts[i] =  0.4*cos(2*pi*(i/n))*w[1] + 0.4*sin(2*pi*(i/n))*w[2] + w[i+2]
    } else if(i == 2){
      ts[i] =  0.4*cos(2*pi*(i/n))*w[2]  + 0.4*sin(2*pi*(i/n))*ts[i-1] + w[i+2]
    } else {
      ts[i] =  0.4*cos(2*pi*(i/n))*ts[i-2] + 0.4*sin(2*pi*(i/n))*ts[i-1] + w[i+2]
    }
  }
  return(ts)
}

generate_AR2s2c1_class2 = function(n, v){
  ts = c()
  w = rnorm(n+2, 0, 1/v)
  
  for(i in 1:n){
    if(i == 1){
      ts[i] =  0.4*cos(2*pi*(i/n))*w[1] + 0.6*sin(2*pi*(i/n))*w[2] + w[i+2]
    } else if(i == 2){
      ts[i] =  0.4*cos(2*pi*(i/n))*w[2]  + 0.6*sin(2*pi*(i/n))*ts[i-1] + w[i+2]
    } else {
      ts[i] =  0.4*cos(2*pi*(i/n))*ts[i-2] + 0.6*sin(2*pi*(i/n))*ts[i-1] + w[i+2]
    }
  }
  return(ts)
}



# Model b

generate_AR2pro_class1 = function(n, v){
  ts = c()
  w = rnorm(n+2, 0, 1/v)
  
  for(i in 1:n){
    if(i == 1){
      ts[i] =  0.6*cos(2*pi*(i/n))*w[1] + 0.4*sin(2*pi*(i/n))*w[2] + w[i+2]
    } else if(i == 2){
      ts[i] =  0.6*cos(2*pi*(i/n))*w[2]  + 0.4*sin(2*pi*(i/n))*ts[i-1] + w[i+2]
    } else {
      ts[i] =  0.6*cos(2*pi*(i/n))*ts[i-2] + 0.4*sin(2*pi*(i/n))*ts[i-1] + w[i+2]
    }
  }
  return(ts)
}

generate_AR2pro_class2 = function(n, v){
  ts = c()
  w = rnorm(n+2, 0, 1/v)
  
  for(i in 1:n){
    if(i == 1){
      ts[i] =  0.4*cos(2*pi*(i/n))*w[1] + 0.6*sin(2*pi*(i/n))*w[2] + w[i+2]
    } else if(i == 2){
      ts[i] =  0.4*cos(2*pi*(i/n))*w[2]  + 0.6*sin(2*pi*(i/n))*ts[i-1] + w[i+2]
    } else {
      ts[i] =  0.4*cos(2*pi*(i/n))*ts[i-2] + 0.6*sin(2*pi*(i/n))*ts[i-1] + w[i+2]
    }
  }
  return(ts)
}


# Model c

generate_AR1s_class1 = function(n, v){
  ts = c()
  x_ini = rnorm(1, 0, 1/v)
  for(i in 1:n){
    if(i == 1){
      ts[i] = 0.5*sin(2*pi*(i/n))*x_ini + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] = 0.5*sin(2*pi*(i/n))*ts[i-1] + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}

generate_AR1s_class2 = function(n, v){
  ts = c()
  x_ini = rnorm(1, 0, 1/v)
  for(i in 1:n){
    if(i == 1){
      ts[i] = 0.5*cos(2*pi*(i/n))*x_ini  + rnorm(1, 0, 1/v) #  
    } else{
      ts[i] = 0.5*cos(2*pi*(i/n))*ts[i-1] + rnorm(1, 0, 1/v) #  
    }
  }
  return(ts)
}

function_names <- c(
  "generate_AR1_class1", "generate_AR1_class2",
  "generate_AR2_class1", "generate_AR2_class2",
  "generate_AR2s2c1_class1", "generate_AR2s2c1_class2",
  "generate_AR2pro_class1", "generate_AR2pro_class2", 
  "generate_AR1s_class1", "generate_AR1s_class2"
)

