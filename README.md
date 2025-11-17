---
title: "ROI Prediction Model"
output:
  github_document:
    toc: true
    toc_depth: 2
---

## Project Overview

This project is a **Return on Investment (ROI) Prediction Model** built in Python using machine learning.  
The aim is to predict the ROI of marketing campaigns using features such as click behavior, impressions, cost, and engagement.  
The model is trained, evaluated, and saved for deployment with Flask.

---

## Tech Stack

- **Python** — Core programming  
- **pandas, NumPy** — Data manipulation & numeric processing  
- **scikit-learn** — Machine learning (regression, scaling, encoding)  
- **joblib** — Model serialization  
- **Flask** — Web app to serve predictions  
- **matplotlib, seaborn** — Visualization of data and results

---

## Workflow / Steps  

```{r}
# 1. Environment Preview (printed in Python script)
# sys, platform, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, flask

# 2. Load & Explore Data  
data <- read.csv("campaign_data.csv")  
str(data)  
summary(data)

# 3. Data Cleaning  
num_cols <- c("Conversion_Rate", "ROI", "Clicks", "Impressions", "Engagement_Score")  
for (col in num_cols) {  
  data[[col]][is.na(data[[col]])] <- mean(data[[col]], na.rm = TRUE)  
}  
cat_cols <- c("Target_Audience", "Customer_Segment")  
for (col in cat_cols) {  
  data[[col]][is.na(data[[col]])] <- as.character(stats::na.omit(data[[col]])[1])  
}

# 4. Feature Engineering  
data$Duration <- as.numeric(gsub(" days", "", data$Duration))  
data$CTR <- data$Clicks / data$Impressions  
data$ROI_per_Cost <- data$ROI / data$Acquisition_Cost

# 5. Encoding  
library(Matrix)  
library(onehot)  # just for example  
encoded <- onehot(data[, cat_cols])  
X_cats <- as.data.frame(predict(encoded, data[, cat_cols]))  
X_nums <- data[, c(num_cols, "CTR", "ROI_per_Cost")]

# 6. Scaling & Model Training  
library(caret)  
preproc <- preProcess(X_nums, method = c("center", "scale"))  
X_scaled <- predict(preproc, X_nums)  
model <- lm(ROI ~ ., data = cbind(X_scaled, X_cats))  
summary(model)

# 7. Evaluation  
pred <- predict(model, cbind(X_scaled, X_cats))  
mse <- mean((data$ROI - pred)^2)  
rsq <- 1 - sum((data$ROI - pred)^2) / sum((data$ROI - mean(data$ROI))^2)  
mse  
rsq
