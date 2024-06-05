# Install necessary packages if not already installed
if (!requireNamespace("quanteda", quietly = TRUE)) {
  install.packages("quanteda")
}
if (!requireNamespace("caret", quietly = TRUE)) {
  install.packages("caret")
}
if (!requireNamespace("quanteda.textmodels", quietly = TRUE)) {
  install.packages("quanteda.textmodels")
}
if (!requireNamespace("matrix", quietly = TRUE)) {
  install.packages("matrix")
}
if (!requireNamespace("e1071", quietly = TRUE)) {
  install.packages("e1071")
}
if (!requireNamespace("ssc", quietly = TRUE)) {
  install.packages("ssc")
}

# Load the required libraries
library(quanteda)
library(quanteda.textmodels)
library(Matrix)
library(caret)
library(e1071)
library(ssc)

# Load the dataset using file.choose() to select the file interactively
reviews <- read.csv(file.choose(), header = TRUE)

# Ensure the dataset has the expected structure
head(reviews)

# Convert the sentiment column to a factor
reviews$sentiment <- factor(reviews$sentiment, levels = c("negative", "positive"))

# Create a corpus and preprocess the text
corpus <- corpus(reviews$review)
tokens <- tokens(corpus, remove_punct = TRUE, remove_numbers = TRUE)
tokens <- tokens_tolower(tokens)
tokens <- tokens_remove(tokens, stopwords("en"))
dfm <- dfm(tokens)

# Ensure the dfm is not too sparse
dfm <- dfm_trim(dfm, min_termfreq = 5)

# Split data into training and test sets
set.seed(123)
trainIndex <- sample(1:nrow(dfm), 0.8 * nrow(dfm))
trainData <- dfm[trainIndex,]
testData <- dfm[-trainIndex,]
trainLabels <- reviews$sentiment[trainIndex]
testLabels <- reviews$sentiment[-trainIndex]

# Further split trainData to simulate labeled and unlabeled data
set.seed(123)
labeledIndex <- sample(1:nrow(trainData), 0.5 * nrow(trainData))
labeledData <- trainData[labeledIndex,]
labeledLabels <- trainLabels[labeledIndex]
unlabeledData <- trainData[-labeledIndex,]
unlabeledLabels <- rep(NA, nrow(unlabeledData))

# Train a Naive Bayes model using labeled data with quanteda.textmodels
inductiveModel <- textmodel_nb(labeledData, labeledLabels)

# Make predictions on the test set
inductivePredictions <- predict(inductiveModel, newdata = testData)
# Evaluate model performance
confusionMatrix(inductivePredictions, testLabels)

#Transductive
# Train a Naive Bayes model using labeled data
transductiveModel <- textmodel_nb(labeledData, labeledLabels)

# Make initial predictions on the unlabeled data
initialPredictions <- predict(transductiveModel, newdata = unlabeledData)

# Combine the labeled and initially predicted labels
combinedData <- rbind(labeledData, unlabeledData)
combinedLabels <- c(labeledLabels, initialPredictions)

# Retrain the model using combined data
transductiveModel <- textmodel_nb(combinedData, combinedLabels)

# Make predictions on the test set
transductivePredictions <- predict(transductiveModel, newdata = testData)

# Evaluate model performance
confusionMatrix(transductivePredictions, testLabels)







