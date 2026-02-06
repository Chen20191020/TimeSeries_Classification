
library(ggplot2)
#  classfication for EEG 

# C3 train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_C3.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_C3.rds")

# Features
ma_diff = c()
for(num_obs in 1:length(X_li_test)){
  res = auto.fit.legen_fast(X_li_test[[num_obs]], c(1:11), c(1:3), method = "LOOCV", inte = T)
  res_coef = res[[1]]
  phi_j = res_coef[[length(res_coef)]]
  ma_diff[num_obs] = max(phi_j) - min(phi_j)
}


C3_feature = ma_diff



# C4 train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_C4.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_C4.rds")

# Features
ma_diff = c()
for(num_obs in 1:length(X_li_test)){
  res = auto.fit.legen_fast(X_li_test[[num_obs]], c(1:11), c(1:3), method = "LOOCV")
  res_coef = res[[1]]
  phi_j = res_coef[[length(res_coef)]]
  ma_diff[num_obs] = max(phi_j) - min(phi_j)
}

C4_feature = ma_diff


# CZ train set and test set 
X_li_train = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/train_set/train_CZ.rds")
X_li_test = readRDS("/Users/chenqian/Desktop/Real_Data(with noise)/test_set/test_CZ.rds")

# Feature
ma_diff = c()
for(num_obs in 1:length(X_li_test)){
  res = auto.fit.legen_fast(X_li_test[[num_obs]], c(1:10), c(1:3), method = "LOOCV", inte = T)
  res_coef = res[[1]]
  phi_j = res_coef[[length(res_coef)]]
  ma_diff[num_obs] = max(phi_j) - min(phi_j)
}

CZ_feature = ma_diff

# Create a data frame
df_feature <- data.frame(
  Value = c(CZ_feature, C3_feature, C4_feature),
  Group = rep(rep(c("Abnormal", "Normal"), each = 75), 3),
  Vector = factor(rep(c("Channel CZ", "Channel C3", "Channel C4"), each = 150), levels = c("Channel CZ", "Channel C3", "Channel C4"))
)
# Plot using ggplot2
ggplot(df_feature, aes(x = Group, y = Value, fill = Group)) +
  geom_boxplot(outlier.shape = NA, width = 0.5) +  
  facet_grid(~ Vector, scales = "free_y", switch = "y") + 
  scale_x_discrete(labels = c("First Half" = "First 75", "Second Half" = "76-150")) +
  scale_y_continuous(limits = c(min(df_feature$Value), max(df_feature$Value)-0.4), 
                     breaks = seq(floor(min(df_feature$Value)), ceiling(max(df_feature$Value)), length.out = 6)) +
  labs(x = "Labels", y = "Value") +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(face = "bold", color = "#993333", size = 15),
    axis.text.y = element_text(face = "bold", color = "#993333", size = 20),
    axis.title.x = element_text(size = 20, face = "bold"),
    axis.title.y = element_text(angle = 90, face = "bold", size = 20),
    panel.grid.major.y = element_line(color = "white", linetype = "solid"),
    panel.grid.major.x = element_blank(),    # Remove vertical grid lines
    panel.background = element_rect(fill = "grey95", color = NA),
    legend.position = "none",  # Remove legends
    strip.text = element_text(face = "bold")  # Bold subtitles (facet labels)
  )

