# Required libraries
library(ggplot2)
library(tibble)



# Data for accuracy and standard deviation

accuracy_data <- tibble::tibble(
  Iteration = rep(c(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4), 11),
  Accuracy = c(
    0.638, 0.812, 0.920, 0.970, 0.990, 0.990, 1.000, 1.000,  # Proposed
    0.607, 0.743, 0.844, 0.912, 0.959, 0.982, 0.994, 0.997,  # LSW
    0.502, 0.544, 0.641, 0.749, 0.857, 0.939, 0.983, 0.995,  # DWT
    0.497, 0.505, 0.499, 0.503, 0.498, 0.504, 0.501, 0.499,  # FLogistic
    0.502, 0.542, 0.662, 0.804, 0.917, 0.972, 0.992, 0.998,  # Rocket
    0.501, 0.501, 0.499, 0.500, 0.502, 0.501, 0.502, 0.512,  # TimeCNN
    0.549, 0.697, 0.843, 0.935, 0.978, 0.993, 0.999, 1.000,  # MultiRocket
    0.503, 0.552, 0.667, 0.813, 0.92,  0.973, 0.992, 0.998,  # Arsenal
    0.500, 0.502, 0.535, 0.592, 0.684, 0.795, 0.898, 0.962,  # Shapelet
    0.554, 0.703, 0.822, 0.906, 0.963, 0.987, 0.996, 0.999,  # STSForest
    0.503, 0.503, 0.500, 0.507, 0.501, 0.502, 0.505, 0.506   # TimeRNN
  ),
  StdDev = c(
    0.067, 0.053, 0.041, 0.025, 0.016, 0.016, 0.010, 0.010,  # Proposed
    0.070, 0.065, 0.051, 0.039, 0.028, 0.018, 0.010, 0.007,  # LSW
    0.070, 0.074, 0.069, 0.061, 0.048, 0.036, 0.019, 0.010,  # DWT
    0.074, 0.070, 0.071, 0.074, 0.072, 0.073, 0.069, 0.070,  # FLogistic
    0.070, 0.070, 0.070, 0.050, 0.040, 0.020, 0.010, 0.010,  # Rocket
    0.025, 0.027, 0.023, 0.027, 0.025, 0.027, 0.037, 0.059,  # TimeCNN
    0.069, 0.063, 0.049, 0.035, 0.020, 0.011, 0.005, 0.002,  # MultiRocket
    0.07, 0.07, 0.066, 0.055, 0.038, 0.024, 0.012, 0.007,    # Arsenal 
    0.072, 0.069, 0.066, 0.071, 0.071, 0.059, 0.044, 0.027,  # Shapelet
    0.074, 0.063, 0.057, 0.043, 0.025, 0.016, 0.009, 0.005,  # STSForest
    0.071, 0.073, 0.074, 0.068, 0.068, 0.072, 0.070, 0.067   # TimeRNN
  ),
  Model = rep(c(
    "Proposed", "LSW", "DWT", "FLogistic", "Rocket", "TimeCNN", 
    "MultiRocket", "Arsenal", "Shapelet", "STSForest", "TimeRNN"
  ), each = 8)
)

# Set distinct colors for each model
model_colors <- c(
  "Proposed" = "red", 
  "LSW" = "#1f78b4", 
  "DWT" = "yellow",
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
  geom_line(linewidth = 0.8) +                                # Thin lines
  geom_point(size = 1.5) +                               # Points
  geom_errorbar(aes(ymin = Accuracy - StdDev, ymax = Accuracy + StdDev), width = 0.005, size = 0.8) +
  scale_x_continuous(limits = c(0.05, 0.4), breaks = seq(0.05, 0.4, by = 0.05)) +
  scale_y_continuous(limits = c(0.2, 1), breaks = seq(0.2, 1, 0.1)) +
  scale_color_manual(values = model_colors, guide = guide_legend(nrow = 2, byrow = TRUE)) +
  labs( x = expression(delta), y = "Accuracy") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", 
    #legend.position = "bottom",         # Legend at the bottom
      legend.box.just = "left",    
        legend.title = element_blank()) +
  theme(
    legend.text = element_text(size = 13, face = "bold"),
    axis.text.x = element_text(face = "bold", color = "#993333", size = 20),
    axis.text.y = element_text(face = "bold", color = "#993333", size = 20),
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




