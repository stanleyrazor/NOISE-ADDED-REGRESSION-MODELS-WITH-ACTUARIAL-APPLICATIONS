
# auto insurance in sweden dataset ----------------------------------------

# dir
setwd("C:/Users/stanley/Desktop/MISCELLANEOUS R/ml projects/regression/anujonthemove-auto-insurance-in-sweden/original")

# libs
library(pacman)
p_load(dplyr, ggplot2, stringr, stringi, ModelMetrics)

# data
autodd <- read.csv("auto_insurance_sweden.csv", as.is=T)

# summary and str
summary(autodd)
glimpse(autodd)
View(autodd)
names(autodd) <- c("claims_no", "total_payment")

# viz
ggplot(data=autodd)+
  geom_point(aes(claims_no, total_payment))+
  labs(title="CLAIMS NUMBER AND TOTAL PAYMENT",
       x="Number of claims",y="Total payment")+
  geom_smooth(aes(claims_no, total_payment), method="lm")

# correlation analysis
cor(autodd$claims_no, autodd$total_payment)

# splitting the data for training and testing
set.seed(29)
test <- sample(63, 10)
dd_train <- autodd[-test,]
dd_test <- autodd[test, ]


# fitting the model
mod1 <- lm(total_payment~claims_no, 
           data=dd_train)
summary(mod1)

# testing the model is a good fit
# test for normality of residuals
shapiro.test(mod1$residuals)

qqnorm(mod1$residuals)
qqline(mod1$residuals, lwd=2, col="blue")

# using the fitted model to predict test
pred_payment <- predict(mod1, dd_test)

# error analysis
mse(pred_payment, dd_test$total_payment)


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
    noisepred[i] <- pred_payment[i] + mean(mod1$residuals[ds[[i]][1:k]])
  }
  
  msevec[k] <- mse(dd_test$total_payment, noisepred)
}

par(mfrow=c(1,2))
# mse plot
plot(x=1:30, y=msevec, type="o", col="maroon", lwd=2, 
     xlab="K values", ylab="Mean square error",
     main="MSE PLOT", 
     sub = "The blue line-Baseline model", col.sub="blue",
     ylim=c(500, 1700))
abline(h=mse(dd_test$total_payment, pred_payment), col="blue", lwd=2)

# rmse plot
plot(x=1:30, y=sqrt(msevec), type="o", col="maroon", lwd=2, 
     xlab="K values", ylab="Rooted Mean square error",
     main="RMSE PLOT", 
     sub = "The blue line-Baseline model", col.sub="blue",
     ylim=c(20, 45))
abline(h=rmse(dd_test$total_payment, pred_payment), col="blue", lwd=2)

