
library(ggplot2)
library(tibble)

# Data for accuracy and standard deviation

accuracy_data <- tibble::tibble(
  Iteration = rep(c(500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500), 9),
  Accuracy = c(
    0.880, 0.902, 0.924, 0.938, 0.948, 0.960, 0.962, 0.968, 0.974, 0.98, 0.98,  # Proposed
    
    0.501, 0.503, 0.506, 0.501, 0.503, 0.5, 0.501, 0.504, 0.5, 0.504, 0.5,  # FLogistic
    0.713, 0.742, 0.759, 0.780, 0.791, 0.802, 0.818, 0.827, 0.843, 0.85, 0.858,  # Rocket
    0.503, 0.505, 0.501, 0.50, 0.508, 0.504, 0.501, 0.506, 0.501, 0.508, 0.51,  # TimeCNN
    0.845, 0.871, 0.893, 0.906, 0.921, 0.932, 0.943, 0.948, 0.957, 0.963, 0.965,  # MultiRocket
    0.72, 0.749, 0.766, 0.776, 0.802, 0.813, 0.823, 0.837, 0.843, 0.854, 0.863,   # Arsenal
    0.561, 0.567, 0.571, 0.576, 0.589, 0.587, 0.588, 0.586, 0.594, 0.595, 0.602,  # Shapelet
    0.824, 0.846, 0.862, 0.882, 0.897, 0.912, 0.921, 0.928, 0.933, 0.94, 0.946,  # STSForest
    0.521, 0.507, 0.504, 0.504, 0.502, 0.501, 0.502, 0.525, 0.499, 0.502, 0.501   # TimeRNN
  ),
  StdDev = c(
    0.048, 0.041, 0.036, 0.034, 0.030, 0.029, 0.027, 0.025, 0.022, 0.022, 0.021,  # Proposed
    0.076, 0.069, 0.073, 0.070, 0.067, 0.073,0.067, 0.074, 0.072, 0.069, 0.07,  # FLogistic
    0.065, 0.064, 0.063, 0.054, 0.058, 0.055, 0.052, 0.055, 0.054, 0.051, 0.046,  # Rocket
    0.061, 0.065, 0.066, 0.065, 0.064, 0.063, 0.064, 0.063, 0.06, 0.067, 0.063,  # TimeCNN
    0.05, 0.049, 0.041, 0.041, 0.039, 0.035, 0.033, 0.032, 0.029, 0.027, 0.026,  # MultiRocket
    0.064, 0.062, 0.058, 0.059, 0.057, 0.053, 0.053, 0.052, 0.054, 0.052, 0.051,  # Arsenal 
    0.073, 0.075, 0.073, 0.072, 0.071, 0.068, 0.072, 0.074, 0.073, 0.076, 0.069,  # Shapelet
    0.054, 0.051, 0.047, 0.048, 0.043, 0.039, 0.038, 0.037, 0.037, 0.032, 0.031,  # STSForest
    0.072, 0.071, 0.068, 0.069, 0.069, 0.069, 0.074, 0.071, 0.074, 0.071, 0.067   # TimeRNN
  ),
  Model = rep(c(
    "Proposed", "FLogistic", "Rocket", "TimeCNN", 
    "MultiRocket", "Arsenal", "Shapelet", "STSForest", "TimeRNN"
  ), each = 11)
)

# Set distinct colors for each model
model_colors <- c(
  "Proposed" = "red", 
  "FLogistic" = "orange", 
  "Rocket" = "green", 
  "TimeCNN" = "blue", 
  "MultiRocket" = "purple", 
  "Arsenal" = "cyan", 
  "Shapelet" = "brown",
  "STSForest" = "darkslateblue",
  "TimeRNN" = "pink"
)

theme_update(plot.title = element_text(hjust = 0.5))
# Generate the plot
ggplot(accuracy_data, aes(x = Iteration, y = Accuracy, color = Model)) +
  geom_line(size = 0.8) +                                # Thin lines
  geom_point(size = 1.5) +                               # Points
  geom_errorbar(aes(ymin = Accuracy - StdDev, ymax = Accuracy + StdDev), width = 0.005, size = 0.8) +
  scale_x_continuous(limits = c(500, 1500), breaks = seq(500, 1500, by = 100)) +
  scale_y_continuous(limits = c(0.2, 1), breaks = seq(0.2, 1, 0.1))  +
  labs( x = expression(n), y = "Accuracy") +
  theme_minimal(base_size = 14) +
  scale_color_manual(values = model_colors) +
  theme(legend.position = "top", 
        #legend.position = "bottom",         # Legend at the bottom
        legend.box.just = "left",    
        legend.title = element_blank()) +
  theme(
    legend.text = element_text(size = 13, face = "bold"),
    axis.text.x = element_text(face = "bold", color = "#993333", size = 17),
    axis.text.y = element_text(face = "bold", color = "#993333", size = 17),
    axis.title.x = element_text(size = 23, face = "bold"),
    axis.title.y = element_text(angle = 90, face = "bold", size = 20),
    #panel.grid.major = element_blank(),    # Remove major grid lines
    #panel.grid.minor = element_blank(),    # Remove minor grid lines
    #legend.title = element_text(face = "bold")
    panel.grid.major.y = element_line(color = "white", linetype = "solid"),
    # panel.grid.major.x = element_line(color = "white", linetype = "solid"),
    panel.grid.major.x = element_line(color = "white", linetype = "solid"),    # Remove vertical grid lines
    panel.background = element_rect(fill = "grey95", color = NA) 
  )

# SIMle::homo.test

