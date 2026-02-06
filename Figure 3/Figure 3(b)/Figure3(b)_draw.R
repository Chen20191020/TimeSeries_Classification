

library(ggplot2)
# Load required libraries
library(tibble)

# Data Preparation

accuracy_data <- tibble::tibble(
  Iteration = rep(c(1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5), 11),
  Accuracy = c(
    # Proposed
    0.97, 0.97, 0.97, 0.968, 0.964, 0.966, 0.963, 0.963, 0.961,
    # LWS
    0.909, 0.911, 0.914,0.909, 0.906, 0.908, 0.91, 0.906, 0.904,
    # DWT
    0.739, 0.744, 0.749, 0.748, 0.747, 0.742, 0.743, 0.739, 0.741,
    # Flrg
    0.497, 0.501, 0.506, 0.499, 0.503, 0.499, 0.502, 0.498, 0.499,
    # Rocket
    0.774, 0.742, 0.702, 0.669, 0.650, 0.634, 0.616, 0.602, 0.593,
    # TimeCNN
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    # MultiRoket
    0.916, 0.909, 0.905, 0.893, 0.888, 0.887, 0.875, 0.869, 0.862,
    # Arsenal
    0.772, 0.738, 0.705, 0.666, 0.637, 0.620, 0.602, 0.583, 0.569,
    # Shaplet
    0.572, 0.566, 0.545, 0.534, 0.521, 0.513, 0.51, 0.507, 0.503,
    
    #STSForest 
    0.891, 0.886, 0.882, 0.874, 0.86, 0.845, 0.844, 0.833, 0.822,
    
    # TimeRNN
    0.508, 0.503, 0.506, 0.513, 0.505, 0.506, 0.506, 0.513, 0.51
  ),
  StdDev = c(
    # Proposed
    0.027, 0.026, 0.028, 0.027, 0.028, 0.028, 0.028, 0.028, 0.030,
    # LSW
    0.043, 0.040, 0.039, 0.041, 0.042, 0.041, 0.042, 0.043, 0.044,
    # DWT
    0.061, 0.061, 0.062, 0.064, 0.062, 0.064, 0.060, 0.061, 0.062,
    # Flrg
    0.073, 0.069, 0.068, 0.071, 0.070, 0.072, 0.072, 0.067, 0.073,
    # Rocket
    0.059, 0.057, 0.054, 0.053, 0.051, 0.048, 0.046, 0.043, 0.040,
    # TimeCNN
    0.02, 0.002, 0, 0, 0, 0, 0, 0, 0,
    # MultiRoket
    0.04, 0.039, 0.04, 0.04, 0.041, 0.043, 0.0476, 0.047, 0.046,
    # Arsenal
    0.061, 0.056, 0.055, 0.051, 0.048, 0.043, 0.046, 0.043, 0.04,
    # Shaplet
    0.075, 0.063, 0.055, 0.041, 0.033, 0.028, 0.023, 0.017, 0.013,
    #STSForest 
    0.046, 0.046, 0.046, 0.048, 0.049, 0.048, 0.048, 0.05, 0.05,
    # TimeRNN
    0.069, 0.070, 0.066, 0.050, 0.065, 0.059, 0.059, 0.054, 0.053
  ),
  Model = rep(c(
    "Proposed", "LSW", "DWT", "FLogistic", "Rocket", "TimeCNN", 
    "MultiRocket", "Arsenal", "Shapelet", "STSForest", "TimeRNN"
  ), each = 9)
)

# Set distinct colors
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

# Plot


theme_update(plot.title = element_text(hjust = 0.5))
# Generate the plot
ggplot(accuracy_data, aes(x = Iteration, y = Accuracy, color = Model)) +
  geom_line(size = 0.8) +                                # Thin lines
  geom_point(size = 1.5) +                               # Points
  geom_errorbar(aes(ymin = Accuracy - StdDev, ymax = Accuracy + StdDev), width = 0.005, size = 0.8) +
  scale_x_continuous(limits = c(1, 5), breaks = seq(1, 5, 0.5)) +
  scale_y_continuous(limits = c(0.25, 1), breaks = seq(0.25, 1, 0.1)) +
  scale_color_manual(values = model_colors, guide = guide_legend(nrow = 2, byrow = TRUE)) +
  labs(x = expression(kappa), y = "Accuracy") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", legend.title = element_blank()) +
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
    panel.grid.major.x = element_blank(),    # Remove vertical grid lines
    panel.background = element_rect(fill = "grey95", color = NA) 
  )

