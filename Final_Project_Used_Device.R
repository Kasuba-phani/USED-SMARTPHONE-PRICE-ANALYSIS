data <- read.csv("C:/Users/kasub/Desktop/CSDA - 6010- Analytics practicum/used_device_data.csv")
head(data)

summary(data)
str(data)
sum(is.na(data))

# Check missing values column-wise
colSums(is.na(data))

# Percentage of missing values per column
colMeans(is.na(data)) * 100

# View rows with missing values
data[!complete.cases(data), ]

#Remove rows with missing values
data_clean <- na.omit(data)
sum(data == 0, na.rm = TRUE)
colSums(data == 0, na.rm = TRUE)
colMeans(data == 0, na.rm = TRUE) * 100
data[rowSums(data == 0, na.rm = TRUE) > 0, ]

data <- data_clean

# Categorical variables
barplot(table(data$device_brand), main="Device Brand Distribution", las=2, col="skyblue")
barplot(table(data$os), main="Operating System Distribution", col="lightgreen")
barplot(table(data$X4g), main="4G Support", col="orange")
barplot(table(data$X5g), main="5G Support", col="purple")

hist(data$screen_size, main="Screen Size Distribution", xlab="Inches", col="skyblue", border="black")

hist(data$rear_camera_mp, main="Rear Camera MP Distribution", xlab="Megapixels", col="lightgreen", border="black")

hist(data$front_camera_mp, main="Front Camera MP Distribution", xlab="Megapixels", col="pink", border="black")

hist(data$internal_memory, main="Internal Memory Distribution", xlab="GB", col="orange", border="black")

hist(data$ram, main="RAM Distribution", xlab="GB", col="purple", border="black")

hist(data$battery, main="Battery Capacity Distribution", xlab="mAh", col="blue", border="black")

hist(data$weight, main="Weight Distribution", xlab="Grams", col="brown", border="black")

hist(data$release_year, main="Release Year Distribution", xlab="Year", col="gray", border="black")

hist(data$days_used, main="Days Used Distribution", xlab="Days", col="lightblue", border="black")

hist(data$normalized_used_price, main="Normalized Used Price Distribution", xlab="Price", col="darkgreen", border="black")

hist(data$normalized_new_price, main="Normalized New Price Distribution", xlab="Price", col="darkblue", border="black")

library(corrplot)
numeric_data <- data[, sapply(data, is.numeric)]

# Compute correlation matrix
cor_matrix <- cor(numeric_data, use = "pairwise.complete.obs")
print(cor_matrix)

# Visualize correlation matrix
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, 
         addCoef.col = "black", # adds correlation values
         diag = FALSE, 
         number.cex = 0.6)


library(ggplot2)

# Screen Size vs Used Price
ggplot(data, aes(x = screen_size, y = normalized_used_price)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Screen Size vs Used Price",
       x = "Screen Size (inches)", 
       y = "Normalized Used Price") +
  theme_minimal()

# Rear Camera vs Used Price
ggplot(data, aes(x = rear_camera_mp, y = normalized_used_price)) +
  geom_point(alpha = 0.5, color = "darkgreen") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Rear Camera MP vs Used Price",
       x = "Rear Camera (MP)", 
       y = "Normalized Used Price") +
  theme_minimal()

# Front Camera vs Used Price
ggplot(data, aes(x = front_camera_mp, y = normalized_used_price)) +
  geom_point(alpha = 0.5, color = "purple") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Front Camera MP vs Used Price",
       x = "Front Camera (MP)", 
       y = "Normalized Used Price") +
  theme_minimal()

# Internal Memory vs Used Price
ggplot(data, aes(x = internal_memory, y = normalized_used_price)) +
  geom_point(alpha = 0.5, color = "orange") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Internal Memory vs Used Price",
       x = "Internal Memory (GB)", 
       y = "Normalized Used Price") +
  theme_minimal()


# RAM vs Used Price
ggplot(data, aes(x = ram, y = normalized_used_price)) +
  geom_point(alpha = 0.5, color = "brown") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "RAM vs Used Price",
       x = "RAM (GB)", 
       y = "Normalized Used Price") +
  theme_minimal()

# Battery vs Used Price
ggplot(data, aes(x = battery, y = normalized_used_price)) +
  geom_point(alpha = 0.5, color = "skyblue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Battery vs Used Price",
       x = "Battery Capacity (mAh)", 
       y = "Normalized Used Price") +
  theme_minimal()

# Weight vs Used Price
ggplot(data, aes(x = weight, y = normalized_used_price)) +
  geom_point(alpha = 0.5, color = "darkorange") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Weight vs Used Price",
       x = "Weight (grams)", 
       y = "Normalized Used Price") +
  theme_minimal()

# Release Year vs Used Price
ggplot(data, aes(x = release_year, y = normalized_used_price)) +
  geom_point(alpha = 0.5, color = "darkred") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Release Year vs Used Price",
       x = "Release Year", 
       y = "Normalized Used Price") +
  theme_minimal()

# Days Used vs Used Price
ggplot(data, aes(x = days_used, y = normalized_used_price)) +
  geom_point(alpha = 0.5, color = "navy") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Days Used vs Used Price",
       x = "Days Used", 
       y = "Normalized Used Price") +
  theme_minimal()

# New Price vs Used Price
ggplot(data, aes(x = normalized_new_price, y = normalized_used_price)) +
  geom_point(alpha = 0.5, color = "darkblue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "New Price vs Used Price",
       x = "Normalized New Price", 
       y = "Normalized Used Price") +
  theme_minimal()

# Device Brand vs Used Price
ggplot(data, aes(x = device_brand, y = normalized_used_price, fill = device_brand)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Device Brand vs Used Price", 
       x = "Device Brand", 
       y = "Normalized Used Price")

# Operating System vs Used Price
ggplot(data, aes(x = os, y = normalized_used_price, fill = os)) +
  geom_boxplot() +
  labs(title = "Operating System vs Used Price", 
       x = "Operating System", 
       y = "Normalized Used Price")

# 4G vs Used Price
ggplot(data, aes(x = as.factor(X4g), y = normalized_used_price, fill = as.factor(X4g))) +
  geom_boxplot() +
  labs(title = "4G Support vs Used Price", 
       x = "4G Support (0/1)", 
       y = "Normalized Used Price")

# 5G vs Used Price
ggplot(data, aes(x = as.factor(X5g), y = normalized_used_price, fill = as.factor(X5g))) +
  geom_boxplot() +
  labs(title = "5G Support vs Used Price", 
       x = "5G Support (0/1)", 
       y = "Normalized Used Price")



## Linear Regression
data$device_brand <- as.factor(data$device_brand)
data$os           <- as.factor(data$os)
data$X4g          <- as.factor(data$X4g)
data$X5g          <- as.factor(data$X5g)

set.seed(42)
idx  <- sample(seq_len(nrow(data)), size = 0.8 * nrow(data))
train <- data[idx, ]
test  <- data[-idx, ]

lm_fit <- lm(normalized_used_price ~ ., data = train)
pred <- predict(lm_fit, newdata = test)

rss  <- sum((test$normalized_used_price - pred)^2)
tss  <- sum((test$normalized_used_price - mean(test$normalized_used_price))^2)
r2   <- 1 - rss/tss
rmse <- sqrt(mean((test$normalized_used_price - pred)^2))

cat("Test R²: ", round(r2, 3), "\n")
cat("Test RMSE:", round(rmse, 3), "\n")

##Stepwise Regression
full_formula <- as.formula("normalized_used_price ~ .")
null_formula <- as.formula("normalized_used_price ~ 1")

full_lm <- lm(full_formula, data = train)

step_lm <- step(
  object = full_lm,
  scope  = list(lower = null_formula, upper = full_formula),
  direction = "both",
  trace = 0
)

pred <- predict(step_lm, newdata = test)

rss  <- sum((test$normalized_used_price - pred)^2)
tss  <- sum((test$normalized_used_price - mean(test$normalized_used_price))^2)
r2   <- 1 - rss/tss
rmse <- sqrt(mean((test$normalized_used_price - pred)^2))

cat("Selected model formula:\n")
print(formula(step_lm))
cat("\nTest R²: ", round(r2, 3), "\n")
cat("Test RMSE:", round(rmse, 3), "\n")

## Random Forest Regression
library(randomForest)
rf_fit <- randomForest(
  normalized_used_price ~ ., 
  data = train,
  ntree = 500,        
  importance = TRUE
)

pred <- predict(rf_fit, newdata = test)

rss  <- sum((test$normalized_used_price - pred)^2)
tss  <- sum((test$normalized_used_price - mean(test$normalized_used_price))^2)
r2   <- 1 - rss/tss
rmse <- sqrt(mean((test$normalized_used_price - pred)^2))

cat("Random Forest Test R²: ", round(r2, 3), "\n")
cat("Random Forest Test RMSE:", round(rmse, 3), "\n")

## Logistic Regression
library(caret)

cutoff <- median(data$normalized_used_price, na.rm = TRUE)
data$price_high <- ifelse(data$normalized_used_price >= cutoff, "1", "0")
data$price_high <- factor(data$price_high, levels = c("0", "1"))

model_data <- subset(data, select = -normalized_used_price)

set.seed(42)
n <- nrow(model_data)
idx <- sample(seq_len(n), size = n)   # shuffle rows

train_end <- floor(0.6 * n)
val_end   <- floor(0.8 * n)

train <- model_data[idx[1:train_end], ]
val   <- model_data[idx[(train_end + 1):val_end], ]
test  <- model_data[idx[(val_end + 1):n], ]
logit_fit <- glm(price_high ~ ., data = train, family = binomial)


#Predict on test set
prob <- predict(logit_fit, newdata = test, type = "response")
pred_class <- ifelse(prob >= 0.5, "1", "0")
pred_class <- factor(pred_class, levels = c("0", "1"))

#Evaluate with confusionMatrix
cm <- confusionMatrix(
  data = pred_class,
  reference = test$price_high,
  positive = "1"
)
print(cm)



## Decision Tree
library(rpart)
library(rpart.plot)


tree_fit <- rpart(
  price_high ~ .,
  data = train,
  method = "class",
  control = rpart.control(cp = 0.001, minsplit = 20, xval = 10)
)

opt_cp <- tree_fit$cptable[which.min(tree_fit$cptable[, "xerror"]), "CP"]
tree_fit_pruned <- prune(tree_fit, cp = opt_cp)

rpart.plot(tree_fit_pruned, type = 2, extra = 106, fallen.leaves = TRUE,
           main = "Decision Tree for Used Smartphone Price Tier")


pred_class <- predict(tree_fit_pruned, newdata = test, type = "class")

cm <- confusionMatrix(
  data = pred_class,
  reference = test$price_high,
  positive = "1"
)

print(cm)


rf_clf <- randomForest(
  price_high ~ .,
  data = train,
  ntree = 500,
  importance = TRUE
)

print(rf_clf)
importance(rf_clf)
varImpPlot(rf_clf, main = "Random Forest - Variable Importance")

#Predict on test set
pred_class <- predict(rf_clf, newdata = test, type = "class")


#Evaluate with confusionMatrix
cm <- confusionMatrix(
  data = pred_class,
  reference = test$price_high,
  positive = "1"
)
print(cm)