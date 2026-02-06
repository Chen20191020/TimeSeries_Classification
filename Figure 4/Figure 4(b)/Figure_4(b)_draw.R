
library(ggplot2)
# Define colors for each method
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

# Create the data frame
data <- data.frame(
  Method = factor(c("Proposed", "MultiRocket", "Rocket", "Arsenal", "STSForest", 
                    "Shapelet", "DWT", "LSW", "FLogistic", "TimeCNN", "TimeRNN"),
                  levels = names(model_colors)),  # Ensure levels match color keys
  CZ = c(94.7, 78.7, 79.3, 78.0, 75.3, 72.7, 73.3, 64.0, 58.7, 55.3, 53.3),
  C3 = c(74.7, 74.0, 70.7, 72.7, 65.3, 64.0, 61.3, 63.3, 52.0, 50.0, 54.7),
  C4 = c(72.6, 76.7, 68.0, 73.3, 71.3, 66.7, 64.7,  58.6, 54.7, 53.3, 52.7),
  Majority = c(85.3, 76.7, 76.7, 76.7, 73.3, 72.7, 65.3, 66.0, 57.3, 54.7, 48.7)
)

theme_update(plot.title = element_text(hjust = 0.5))
# Create bar plot with specific colors
ggplot(data, aes(x = reorder(Method, -Majority), y = Majority, fill = Method)) +
  geom_bar(stat = "identity", width = 0.3, show.legend = FALSE) +
  scale_fill_manual(values = model_colors, guide = "none") +  # Avoid legend duplication
  theme_minimal() +
  labs(
    x = "Method",
    y = "Accuracy"
  ) + theme(
    legend.text = element_text(size = 13, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold", color = "#993333", size = 12),
    axis.text.y = element_text(face = "bold", color = "#993333", size = 12),
    axis.title.x = element_text(size = 15, face = "bold"),
    axis.title.y = element_text(angle = 90, face = "bold", size = 15),
    #panel.grid.major = element_blank(),    # Remove major grid lines
    #panel.grid.minor = element_blank(),    # Remove minor grid lines
    #legend.title = element_text(face = "bold")
    panel.grid.major.y = element_line(color = "white", linetype = "solid"),
    # panel.grid.major.x = element_line(color = "white", linetype = "solid"),
    panel.grid.major.x = element_line(color = "white", linetype = "solid"),    # Remove vertical grid lines
    panel.background = element_rect(fill = "grey95", color = NA) 
  ) 

