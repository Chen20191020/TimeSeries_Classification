
library(wavethresh)

# Model 7 (5,2), (4,2)
generate_lsw1_class1 = function(n, v = 1){
  x = seq(0,1, length.out = n)
  v9 = (0.7*sin(2*pi*x))^2   
  v8 = (0.6*cos(2*pi*x))^2 
  myspec = cns(n, filter.number =  1)
  myspec = putD(myspec, level = 9, v = v9)
  myspec = putD(myspec, level = 8, v = v8)
  myLSWproc = LSWsim(myspec)
  return(myLSWproc)
}

generate_lsw1_class2 = function(n, v = 1){
  x = seq(0,1, length.out = n)
  v9 = (0.7*sin(2*pi*x))^2  
  v8 = (0.5*cos(2*pi*x))^2  
  myspec = cns(n, filter.number =  1)
  myspec = putD(myspec, level = 9, v = v9)
  myspec = putD(myspec, level = 8, v = v8)
  myLSWproc = LSWsim(myspec)
  return(myLSWproc)
}


# Model 8 (5,3) (4,3)

generate_lsw2_class1 = function(n, v = 1){

  x = seq(0,1, length.out = n)
  v9 = (0.86*sin(2*pi*x))^2   
  v8 = (0.9*exp((x-1/2)^2))^2 
  myspec = cns(n, filter.number =  2)
  myspec = putD(myspec, level = 9, v = v9)
  myspec = putD(myspec, level = 8, v = v8)
  myLSWproc = LSWsim(myspec)
  return(myLSWproc)
}

generate_lsw2_class2 = function(n, v = 1){
 
  x = seq(0,1, length.out = n)
  v9 = (sin(2*pi*x))^2   
  v8 = (0.9*exp((x-1/2)^2))^2  
  myspec = cns(n, filter.number =  2)
  myspec = putD(myspec, level = 9, v = v9)
  myspec = putD(myspec, level = 8, v = v8)
  myLSWproc = LSWsim(myspec)
  return(myLSWproc)
}



# Model 9 case 3 
generate_lsw3_class1 = function(n, v = 1){
  x = seq(0,1, length.out = n)
  v9 = (0.2*sin(2*pi*x))^2   
  myspec = cns(n, filter.number =  1)
  myspec = putD(myspec, level = 9, v = v9)
  myLSWproc = LSWsim(myspec)
  return(myLSWproc)
}

generate_lsw3_class2 = function(n, v = 1){
  x = seq(0,1, length.out = n)
  v9 = (0.4*sin(2*pi*x))^2   
  myspec = cns(n, filter.number =  1)
  myspec = putD(myspec, level = 9, v = v9)
  myLSWproc = LSWsim(myspec)
  return(myLSWproc)
}

function_names <- c(
  "generate_lsw1_class1", "generate_lsw1_class2",
  "generate_lsw2_class1", "generate_lsw2_class2",
  "generate_lsw3_class1", "generate_lsw3_class2"
)
