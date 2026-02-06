library(ggplot2)


accuracy_data <- tibble::tibble(
  Iteration = rep(c(256, 512, 1024, 2048, 4096), 2),
  Accuracy = c(
    0.58, 0.65, 0.75, 0.86, 0.95,  # DWT
    0.73, 0.82, 0.91, 0.98, 1.00  # LSW
  ),
  StdDev = c(
    0.07, 0.07, 0.06, 0.05, 0.03,  # DWT
    0.06, 0.06, 0.04, 0.02, 0.01  # LSW
    
  ),
  Model = rep(c(
    "DWT", "LSW"
  ), each = 5)
)

# Set distinct colors for each model
model_colors <- c(
  "DWT" = "red", 
  "LSW" = "blue"
  
)

theme_update(plot.title = element_text(hjust = 0.5))
# Generate the plot
ggplot(accuracy_data, aes(x = Iteration, y = Accuracy, color = Model)) +
  geom_line(size = 0.8) +                                # Thin lines
  geom_point(size = 1.5) +                               # Points
  geom_errorbar(aes(ymin = Accuracy - StdDev, ymax = Accuracy + StdDev), width = 0.005, size = 0.8) +
  scale_x_continuous(trans = "log2", limits = c(256, 4096), breaks = c(256, 512, 1024, 2048, 4096)) +
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


