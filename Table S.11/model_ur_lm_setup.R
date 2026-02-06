

# situation: our method has no way to do this. 
# Model Unit root
generate_MAill_class1 = function(n, v){
  ts = c()
  epsilon = rnorm(n+2, 0, 1/v)
  
  for(i in 3:(n+2)){
    t = i/n
    ts[i-2] = 0.95 * sin(2 * pi * t)*epsilon[i-2]   + epsilon[i] #  
  }
  return(ts)
}

generate_MAill_class2 = function(n, v){
  ts = c()
  epsilon = rnorm(n+2, 0, 1/v)
  
  for(i in 3:(n+2)){
    t = i/n
    ts[i-2] = 0.85 * sin(2 * pi * t)*epsilon[i-2]   + epsilon[i] #  
  }
  return(ts)
}

# Model Long memory

# Function to simulate Time-Varying ARFIMA(0, d(u), 0)
# Returns: A numeric vector of the simulated time series
generate_tv_arfima = function(n, type = "A", K = 200) {
  # n    : Series length
  # type : "A" for monotone drift, "B" for smooth regime change
  # K    : Truncation limit for the filter (lag depth)
  
  # 1. Setup Time and Scale
  t = 1:n
  u = t / n
  
  # 2. Define d(u) based on the options
  if (type == "A") {
    d_vals = 0.1 + 0.35 * u
  } else if (type == "B") {
    d_vals = 0.25 + 0.15 * sin(2 * pi * u)
  } else {
    stop("Invalid type. Choose 'A' or 'B'.")
  }
  
  # 3. Generate Innovations (epsilon)
  # Generate extra noise to handle the startup lag
  eps = rnorm(n + K, mean = 0, sd = 1) 
  
  # 4. Simulation Loop (Time-Varying Filter)
  X = numeric(n)
  
  for (i in 1:n) {
    current_d = d_vals[i]
    
    # Calculate coefficients psi_k for k = 0 to K
    # Formula: Gamma(k + d) / (Gamma(d) * Gamma(k + 1))
    k_seq = 0:K
    #psi = gamma(k_seq + current_d)/(gamma(current_d) * gamma(k_seq + 1))
    log_psi = lgamma(k_seq + current_d) - lgamma(current_d) - lgamma(k_seq + 1)
    psi = exp(log_psi)
    
    # Extract the relevant window of innovations
    # Corresponds to eps indices from (i + K) down to i
    eps_window = eps[(i + K):(i)] 
    
    # Compute dot product
    X[i] = sum(psi * eps_window)
  }
  
  # Return only the vector
  return(X)
}


generate_ARFIMA_class1 = function(n, v){
  ts = generate_tv_arfima(n = n, type = "A")
  return(ts)
}

generate_ARFIMA_class2 = function(n, v){
  ts = generate_tv_arfima(n = n, type = "B")
  return(ts)
}


function_names <- c(
  "generate_MAill_class1", "generate_MAill_class2",
  "generate_ARFIMA_class1", "generate_ARFIMA_class2"
)

