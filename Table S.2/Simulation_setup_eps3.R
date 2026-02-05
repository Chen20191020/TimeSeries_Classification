# All regression cases with Epsilon 3

# case 1 (with Epsilon 3)
generate_AR1_class1 = function(n, v){
  ts = c()
  x_ini = rnorm(1,0,1)
  for(i in 1:n){
    if(i == 1){
      ts[i] = 0.2*cos(2*pi*(i/n))*x_ini  + (v/2 + v*i/n*(1/2))*rnorm(1, 0, 1)  
    } else{
      ts[i] = 0.2*cos(2*pi*(i/n))*ts[i-1] + (v/2 + v*i/n*(1/2))*rnorm(1, 0, 1)   
    }
  }
  return(ts)
}

generate_AR1_class2 = function(n, v){
  ts = c()
  x_ini = rnorm(1,0,1)
  for(i in 1:n){
    if(i == 1){
      ts[i] = 0.4*cos(2*pi*(i/n))*x_ini + (v/2 + v*i/n*(1/2))*rnorm(1, 0, 1)   
    } else{
      ts[i] = 0.4*cos(2*pi*(i/n))*ts[i-1] + (v/2 + v*i/n*(1/2))*rnorm(1, 0, 1)   
    }
  }
  return(ts)
}


# case 2

generate_AR2_class1 = function(n, v){
  ts = c()
  w = rnorm(n+2, 0, 1) 
  
  for(i in 1:n){
    if(i == 1){
      ts[i] =  (0.6*sin(2*pi*(i/n)))*w[1] + 0.4*w[2] + (v/2 + v*i/n*(1/2))*w[i+2]
    } else if(i == 2){
      ts[i] =  (0.6*sin(2*pi*(i/n)))*w[2]  + 0.4*ts[i-1] + (v/2 + v*i/n*(1/2))*w[i+2]
    } else {
      ts[i] =  (0.6*sin(2*pi*(i/n)))*ts[i-2] + 0.4*ts[i-1] + (v/2 + v*i/n*(1/2))*w[i+2]
    }
  }
  return(ts)
}

generate_AR2_class2 = function(n, v){
  ts = c()
  w = rnorm(n+2, 0, 1)
  
  for(i in 1:n){
    if(i == 1){
      ts[i] =  0.4*cos(2*pi*(i/n))*w[1] + 0.6*w[2] + (v/2 + v*i/n*(1/2))*w[i+2]
    } else if(i == 2){
      ts[i] =  0.4*cos(2*pi*(i/n))*w[2]  + 0.6*ts[i-1] + (v/2 + v*i/n*(1/2))*w[i+2]
    } else {
      ts[i] =  0.4*cos(2*pi*(i/n))*ts[i-2] + 0.6*ts[i-1] + (v/2 + v*i/n*(1/2))*w[i+2]
    }
  }
  return(ts)
}

# Case 3
generate_nAR1_class1 = function(n, v){
  ts = c()
  x_ini = rnorm(1,0,1)
  for(i in 1:n){
    if(i == 1){
      ts[i] =  0.4*(cos(2*pi*(i/n))+1)*x_ini  + (v/2 + v*i/n*(1/2))*rnorm(1, 0, 1)  
    } else{
      ts[i] =  0.4*(cos(2*pi*(i/n))+1)*ts[i-1] + (v/2 + v*i/n*(1/2))*rnorm(1, 0, 1)   
    }
  }
  return(ts)
}

generate_MA2_class2 = function(n, v){
  ts = c()
  epsilon = rnorm(n+2, 0, 1)
  
  for(i in 3:(n+2)){
    ts[i-2] = 0.3*(v/2 + v*(i-2)/n*(1/2))*epsilon[i-2] + 0.4*(v/2 + v*(i-1)/n*(1/2))*epsilon[i-1]  + (v/2 + v*i/n*(1/2))*epsilon[i] #  
  }
  return(ts)
}


# case 4

generate_nAR1.2_class1 = function(n, v){
  ts = c()
  x_ini = rnorm(1,0,1)
  for(i in 1:n){
    if(i == 1){
      ts[i] =  (1.5*sin(2*pi*(i/n)))*exp(-(i/n)*x_ini^2)  + (v/2 + v*i/n*(1/2))*rnorm(1, 0, 1)  
    } else{
      ts[i] = (1.5*sin(2*pi*(i/n)))*exp(-(i/n)*ts[i-1]^2) + (v/2 + v*i/n*(1/2))*rnorm(1, 0, 1)   
    }
  }
  return(ts)
}

generate_nAR1.2_class2 = function(n, v){
  ts = c()
  x_ini = rnorm(1,0,1)
  for(i in 1:n){
    if(i == 1){
      ts[i] =  (0.5*cos(2*pi*(i/n)))*exp(-(i/n)*-x_ini^2)  + (v/2 + v*i/n*(1/2))*rnorm(1, 0, 1)   
    } else{
      ts[i] =  (0.5*cos(2*pi*(i/n)))*exp(-(i/n)*ts[i-1]^2) + (v/2 + v*i/n*(1/2))*rnorm(1, 0, 1)   
    }
  }
  return(ts)
}

# case 5

generate_AR2.2_class1 = function(n, v){
  ts = c()
  w = rnorm(n+2, 0, 1) 
  
  for(i in 1:n){
    if(i == 1){
      ts[i] =  0.2*w[1] + 0.2*sin(2*pi*(i/n))*w[2] + (v/2 + v*i/n*(1/2))*w[i+2]
    } else if(i == 2){
      ts[i] =  0.2*w[2]  + 0.2*sin(2*pi*(i/n))*ts[i-1] + (v/2 + v*i/n*(1/2))*w[i+2]
    } else {
      ts[i] =  0.2*ts[i-2] + 0.2*sin(2*pi*(i/n))*ts[i-1] + (v/2 + v*i/n*(1/2))*w[i+2]
    }
  }
  return(ts)
}

generate_AR2.2_class2 = function(n, v){
  ts = c()
  w = rnorm(n+2, 0, 1)
  
  for(i in 1:n){
    if(i == 1){
      ts[i] =  (0.2*sin(2*pi*(i/n)))*w[1] + 0.2*w[2] + (v/2 + v*i/n*(1/2))*w[i+2]
    } else if(i == 2){
      ts[i] =  (0.2*sin(2*pi*(i/n)))*w[2]  + 0.2*ts[i-1] + (v/2 + v*i/n*(1/2))*w[i+2]
    } else {
      ts[i] =  (0.2*sin(2*pi*(i/n)))*ts[i-2] + 0.2*ts[i-1] + (v/2 + v*i/n*(1/2))*w[i+2]
    }
  }
  return(ts)
}



# case 6
generate_nAR2_class1 = function(n, v){
  ts = c()
  w = rnorm(n+2, 0, 1) / v
  
  for(i in 1:n){
    if(i == 1){
      ts[i] =  0.2*exp(-(i/n)*w[1]^2) + 0.2*(sin(2*pi*(i/n))+1)*(1/(w[2]+1)) + (v/2 + v*i/n*(1/2))*w[i+2]
    } else if(i == 2){
      ts[i] =  0.2*exp(-(i/n)*w[2]^2)  + 0.2*(sin(2*pi*(i/n))+1)*(1/(ts[i-1]+1)) + (v/2 + v*i/n*(1/2))*w[i+2]
    } else {
      ts[i] =  0.2*exp(-(i/n)*ts[i-2]^2) + 0.2*(sin(2*pi*(i/n))+1)*(1/(ts[i-1]+1)) + (v/2 + v*i/n*(1/2))*w[i+2]
    }
  }
  return(ts)
}

generate_nAR2_class2 = function(n, v){
  ts = c()
  w = rnorm(n+2, 0, 1) / v
  for(i in 1:n){
    if(i == 1){
      ts[i] =  0.3*(sin(2*pi*(i/n))+1)*w[1] + 0.2*exp(-(i/n)*w[2]^2) + (v/2 + v*i/n*(1/2))*w[i+2]
    } else if(i == 2){
      ts[i] =  0.3*(sin(2*pi*(i/n))+1)*w[2]  + 0.2*exp(-(i/n)*ts[i-1]^2) + (v/2 + v*i/n*(1/2))*w[i+2]
    } else {
      ts[i] =  0.3*(sin(2*pi*(i/n))+1)*ts[i-2] + 0.2*exp(-(i/n)*ts[i-1]^2) + (v/2 + v*i/n*(1/2))*w[i+2]
    }
  }
  return(ts)
}


function_names <- c(
  "generate_AR1_class1", "generate_AR1_class2",
  "generate_AR2_class1", "generate_AR2_class2",
  "generate_nAR1_class1", "generate_MA2_class2",
  "generate_nAR1.2_class1", "generate_nAR1.2_class2",
  "generate_AR2.2_class1", "generate_AR2.2_class2",
  "generate_nAR2_class1", "generate_nAR2_class2"
)

