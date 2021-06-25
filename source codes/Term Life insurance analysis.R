

# term life assurance analysis --------------------------------------------

setwd("C:/Users/stanley/Desktop/MISCELLANEOUS R/research")

# data
termlife <- read.csv("TermLife.csv", as.is = T)
glimpse(termlife)

# only choosing three variables to explain
# since distribution of FACE and INCOME is much skewed we conside their log

termlife$GENDER <- factor(termlife$GENDER)
termlife$MARSTAT <- factor(termlife$MARSTAT)
termlife$ETHNICITY <- factor(termlife$ETHNICITY)
termlife$SMARSTAT <- factor(termlife$SMARSTAT)
termlife$SGENDER <- factor(termlife$SGENDER)








tl <- select(termlife, EDUCATION, NUMHH, INCOME, FACE)

#log transform
tl$FACE <- ifelse(tl$FACE == 0, yes = NA, no = tl$FACE)
tl <- na.omit(tl)

tl[, 3:4] <- apply(tl[, 3:4], 2, log)

# visualizations
library(GGally)
ggpairs(tl)

# splitting data into test and train
set.seed(29)
test <- sample(275, 50)
dd_train <- tl[-test,]
dd_test <- tl[test, ]

# model
tl_mod <- lm(FACE~., data=dd_train)
summary(tl_mod)


# using the fitted model to predict test
pred_face <- predict(tl_mod, dd_test)

# error analysis
mse(pred_face, dd_test$FACE)


# NOISE ADDED MODELS ------------------------------------------------------

scaletest <- apply(dd_test, 2, scale)
scaletrain <- apply(dd_train, 2, scale)

summary(scaletrain)
summary(scaletest)

# the noise adding, using 30 neighbours
ds <- combine_knnlm(train = scaletrain, 
                    test = scaletest, 
                    k=30)

# starting a loop for one to thirty
msevec <- vector()
noisepred <- vector()

for (k in 1:30)
{
  message(paste("Running batch: ", k))
  
  for (i in 1:nrow(scaletest))
  {
    noisepred[i] <- pred_face[i] + mean(tl_mod$residuals[ds[[i]][1:k]])
  }
  
  msevec[k] <- mse(dd_test$FACE, noisepred)
}

par(mfrow=c(1,2))
# mse plot
plot(x=1:30, y=msevec, type="o", col="maroon", lwd=2, 
     xlab="K values", ylab="Mean square error",
     main="MSE PLOT", 
     sub = "The blue line-Baseline model", col.sub="blue",
     ylim = c(.4, 2.1))
abline(h=mse(dd_test$FACE, pred_face), col="blue", lwd=2)

# rmse plot
plot(x=1:30, y=sqrt(msevec), type="o", col="maroon", lwd=2, 
     xlab="K values", ylab="Rooted Mean square error",
     main="RMSE PLOT", 
     sub = "The blue line-Baseline model", col.sub="blue",
     ylim=c(.6, 1.5))
abline(h=rmse(dd_test$FACE, pred_face), col="blue", lwd=2)

md <- cbind(1:30, msevec) %>%
  as.data.frame()
colnames(md) <- c("Neighbor", "msevec")

p <- ggplot(data=md)+
  geom_line(aes(x=(Neighbor), y=msevec), col="blue")+
  geom_point(aes(x=(Neighbor), y=msevec), col="blue")+
  geom_hline(yintercept = mse(pred_face, dd_test$FACE),
             col="red")+
  labs(title="PERFORMANCE OF NOISE ADDED MODELS", 
       x="K parameter in Noise added models",
       y="Mean Squared Error",
       subtitle = "Red-The baseline model")
ggplotly(p)


# Incorporating a categorical variable in the analysis --------------------


# selecting the variables TOTINCOME, SEDUCATION, MARSTAT, GENDER

newtl <- select(termlife, SEDUCATION, TOTINCOME, MARSTAT, GENDER, FACE)
newtl$GENDER <- ifelse(newtl$GENDER == 1, "MALE", "FEMALE")
newtl$MARSTAT <- case_when(newtl$MARSTAT == 1 ~ "MARRIED",
                           newtl$MARSTAT == 2 ~ "PARTNER",
                           newtl$MARSTAT == 0 ~ "OTHER")
newtl$GENDER <- factor(newtl$GENDER)
newtl$MARSTAT <- factor(newtl$MARSTAT)

# splitting data into test and train
set.seed(29)
test <- sample(275, 50)
dd_train <- newtl[-test,]
dd_test <- newtl[test, ]

catmod <- lm(FACE~., data=dd_train)
summary(catmod)

catpred <- predict(catmod, dd_test)
mse(catpred, dd_test$FACE)

# noise adding by splitting the variables into dummy codes

source("C:/Users/stanley/Desktop/MISCELLANEOUS R/random/one hot encoding.R")

marstat <- onehot_swap(newtl$MARSTAT)
gender <- onehot_swap(newtl$GENDER)

final_tl <- cbind(newtl[, 1:2], newtl[, 5], marstat, gender)
colnames(final_tl) <- c("SEDUCATION", "TOTINCOME", "FACE", "MARRIED", 
                        "PARTNER", "OTHER", "MALE", "FEMALE")
set.seed(29)
test <- sample(275, 50)
dd_train <- final_tl[-test,]
dd_test <- final_tl[test, ]




# NOISE ADDED MODELS ------------------------------------------------------

scaletest <- apply(dd_test, 2, scale)
scaletrain <- apply(dd_train, 2, scale)

summary(scaletrain)
summary(scaletest)

# the noise adding, using 30 neighbours
ds <- combine_knnlm(train = scaletrain, 
                    test = scaletest, 
                    k=30)

# starting a loop for one to thirty
msevec <- vector()
noisepred <- vector()

for (k in 1:30)
{
  message(paste("Running batch: ", k))
  
  for (i in 1:nrow(scaletest))
  {
    noisepred[i] <- catpred[i] + mean(catmod$residuals[ds[[i]][1:k]])
  }
  
  msevec[k] <- mse(dd_test$FACE, noisepred)
}

md <- cbind(1:30, msevec) %>%
  as.data.frame()
colnames(md) <- c("Neighbor", "msevec")

p <- ggplot(data=md)+
  geom_line(aes(x=(Neighbor), y=msevec), col="blue")+
  geom_point(aes(x=(Neighbor), y=msevec), col="blue")+
  geom_hline(yintercept = mse(pred_face, dd_test$FACE),
             col="red")+
  labs(title="PERFORMANCE OF NOISE ADDED MODELS", 
       x="K parameter in Noise added models",
       y="Mean Squared Error",
       subtitle = "Red-The baseline model")
ggplotly(p)
