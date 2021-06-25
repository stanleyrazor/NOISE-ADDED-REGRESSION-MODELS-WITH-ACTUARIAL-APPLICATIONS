

# Auto mobile insurance claims --------------------------------------------

# dir
setwd("C:/Users/stanley/Desktop/MISCELLANEOUS R/research")

# data
liq <- read.csv("Liquidity.csv")
glimpse(liq)

liq <- select(liq, VOLUME, NTRAN, AVGT)
summary(liq)

# splitting data into training and testing
# splitting the data for training and testing
set.seed(29)
test <- sample(123, 50)
dd_train <- liq[-test,]
dd_test <- liq[test, ]



liqmod <- lm(VOLUME~NTRAN+AVGT, data=dd_train)
summary(liqmod)

# metric for baseline
# using the fitted model to predict test
pred_vol <- predict(liqmod, dd_test)

# error analysis
mse(pred_vol, dd_test$VOLUME)

# Noise adding on regression
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
    noisepred[i] <- pred_vol[i] + mean(liqmod$residuals[ds[[i]][1:k]])
  }
  
  msevec[k] <- mse(dd_test$VOLUME, noisepred)
}

md <- cbind(1:30, msevec) %>%
  as.data.frame()
colnames(md) <- c("Neighbor", "msevec")
p <- ggplot(data=md)+
  geom_line(aes(x=(Neighbor), y=msevec), col="blue")+
  geom_point(aes(x=(Neighbor), y=msevec), col="blue")+
  geom_hline(yintercept = mse(pred_vol, dd_test$VOLUME),
             col="red")+
  labs(title="PERFORMANCE OF NOISE ADDED MODELS", 
       x="K parameter in Noise added models",
       y="Mean Squared Error",
       subtitle = "Red-The baseline model")
ggplotly(p)

