
# LIFE EXPECTANCY DATA

For the experimental analysis, I considered this first dataset from the World Health Organization (WHO) in conjunction with the United Nations. The dataset which is made freely available for download in the Kaggle repository has an aim of answering the question on factors affecting life expectancy by considering immunization factors, mortality factors, economic factors, social factors and other health related factors as well.

The original dataset includes all the countries, thus I had to filter and get Kenyan data. It is also important to note that in my data analysis, I discovered that the data exhibited multi-collinearity, and thus I had to do away with several variables, while making sure that every category of factors i.e. social, immunization, mortality has at least one variable that ends up in the model.

The data is split into the training and testing set as follows:
  
  - Training set - Data from 2002 to 2011
- Testing set - Data from 2012 to 2015.

After careful Exploratory Data Analysis, I ended up with the following variables:
  
  1. Polio - This variable represents the total proportion of children immunized against Polio in that given year.

2. Adult Mortality - This variable gives the adult mortality rates of that year.

3. Total expenditure - This variable reports the total expenditure of that particular year

4. Gross Domestic Product

5. Alcohol - This variable reports the total proportion of the population which consume alcohol products.

6. Population - This variable gives the total number of people, as per the national census.

7. Life expectancy - This is the dependent variable which we wish to build a model and predict on. The life expectancy is given in years.


## DATA PREPARATION

```{r start}

# dir
setwd("C:/Users/stanley/Desktop/MISCELLANEOUS R/ml projects/regression/life expectancy analysis")

#libs
library(pacman)
p_load(dplyr, ggplot2, stringr, stringi, plotly)

# data
dd <- read.csv("Life Expectancy Data.csv", as.is=T)

# filtering kenyan data
kenya <- dd %>%
  filter(Country == "Kenya")


# omitting character variables such as country name and status
kenya <- kenya[-c(1,3)]

# dealing with missing values
# imputting using mean
impute_mean <- function(c)
{
  x <- ifelse(test = is.na(c),
              yes = mean(c, na.rm = T),
              no = c)
  return(x)
}

kenya <- apply(kenya, 2, impute_mean) %>%
  data.frame()

# creating the train and test dataset
# the test dataset is the years 2015, 2014, 2013, 2012
k_test <- kenya[1:4, ]
k_train <- kenya[-c(1:4), ]


# summary and str of the train dataset
glimpse(k_train)



# only allowing one group per category
# categories
# Immunization related factors - Hepatitis, Polio, Diptheria, Measles
# Mortality factors - Adult mortality, Infant death,  Under five deaths, Life expectancy
# Economical factors - percentage expenditure, total expenditure, GDP, Income composition of resources
# Social factors - Alcohol, BMI, HIV/AIDS, Population, thinness, schooling

# selecting:
# categories
# Immunization related factors - polio
# Mortality factors - Adult mortality,  Under five deaths, Life expectancy
# Economical factors - total expenditure, GDP, Income composition of resources
# Social factors - Alcohol, HIV/AIDS, Population, only 1 thinness, schooling

k_train <- select(k_train, Polio, Adult.Mortality,
                  Life.expectancy, Total.expenditure, GDP, 
                  Alcohol, Population)

k_test <- select(k_test, Polio, Adult.Mortality,
                 Life.expectancy, Total.expenditure, GDP, 
                 Alcohol, Population)
```

## DATA VISUALIZATION

```{r, viz}

# data viz
attach(k_train)

# polio
ggplot(data=k_train)+
  geom_point(aes(Polio, Life.expectancy))+
  labs(title="POLIO VACCINATION ON LIFE EXPECTANCY",
       x="Numer of polio successful vaccinations",
       y="Life expectancy(in years)")

# adult mortality
ggplot(data=k_train)+
  geom_point(aes(Adult.Mortality, Life.expectancy))+
  labs(title="ADULT MORTALITY ON LIFE EXPECTANCY",
       x="Adult Mortality",
       y="Life expectancy(in years)")


# total expenditure
ggplot(data=k_train)+
  geom_point(aes(Total.expenditure, Life.expectancy))+
  labs(title="TOTAL EXPENDITURE ON LIFE EXPECTANCY",
       x="Total expenditure",
       y="Life expectancy(in years)")

# GDP
ggplot(data=k_train)+
  geom_point(aes(GDP, Life.expectancy))+
  labs(title="GDP ON LIFE EXPECTANCY",
       x="Gross Domestic Product",
       y="Life expectancy(in years)")
# SUGGEST THAT 2000 AND 2001 ARE OUTLIER YEARS

# alcohol
ggplot(data=k_train)+
  geom_point(aes(Alcohol, Life.expectancy))+
  labs(title="ALCOHOL ON LIFE EXPECTANCY",
       x="Alcohol",
       y="Life expectancy(in years)")
# suggest quadratic/log
cor(Life.expectancy, Alcohol)
cor(Life.expectancy, log(Alcohol))

# population
ggplot(data=k_train)+
  geom_point(aes(Population, Life.expectancy))+
  labs(title="POPULATION ON LIFE EXPECTANCY",
       x="Population",
       y="Life expectancy(in years)")
```

## MODEL BUILDING

After careful selection of variables, the best fitted regression model is as shown below:
  
  ```{r, model}
# we omit outliers from the data, the outliers are the data for the years 2000 and 2001
k_train1 <- k_train[1:(nrow(k_train)-2), ]
#View(k_train1)

mod3<- lm(Life.expectancy ~ Polio + Adult.Mortality + Total.expenditure +
            GDP + Alcohol + Population,
          data=k_train1)
summary(mod3)
```

The model has majority of its variables as significant and the adjusted R squared metric is .99 indicating that the model is explaining 99% of the total variation.

The mean squared and rooted mean squared metrics for the pure regression model are shown below:
  
  ```{r, msermse}
# using the model to make predictions
library(ModelMetrics)

pred_expectancy <- predict(mod3, k_test)

# assessing accuarcy
message(paste("The MSE is: ", mse(pred_expectancy, k_test$Life.expectancy)))
message(paste("The RMSE is: ", rmse(pred_expectancy, k_test$Life.expectancy)))
```

From now on, we shall treat this regression model as a baseline model.

## NOISE ADDING

```{r, noise adding}
source("C:/Users/stanley/Desktop/MISCELLANEOUS R/ml projects/Knn/knn classifier/combine knn_lm.R")

scaletest <- apply(k_test, 2, scale)
scaletrain <- apply(k_train1, 2, scale)


# the noise adding, using 30 neighbours
ds <- combine_knnlm(train = scaletrain, 
                    test = scaletest, 
                    k=10)

# starting a loop for one to thirty
msevec <- vector()
noisepred <- vector()

for (k in 1:10)
{
  message(paste("Running batch: ", k))
  
  for (i in 1:nrow(scaletest))
  {
    noisepred[i] <- pred_expectancy[i] + mean(mod3$residuals[ds[[i]][1:k]])
  }
  
  msevec[k] <- mse(k_test$Life.expectancy, noisepred)
}

md <- cbind(1:10, msevec) %>%
  as.data.frame()
colnames(md) <- c("Neighbor", "msevec")
p <- ggplot(data=md)+
  geom_line(aes(x=(Neighbor), y=msevec), col="blue", lwd=1.5)+
  geom_hline(yintercept = mse(pred_expectancy, k_test$Life.expectancy),
             col="red", lwd=2)+
  labs(title="PERFORMANCE OF NOISE ADDED MODELS", 
       x="K parameter in Noise added models",
       y="Mean Squared Error",
       subtitle = "Red-The baseline model")
ggplotly(p)


```

In noise adding, since we have only 10 data points in the training dataset, we will try and fit 10 noise added models, using k equals to 1 neighbor, all the way to k equals 10 neighbors, all while recording useful statistics (MSE and RMSE).

As can be seen from the plots, where the blue curves represent the MSE  of the noise added models, while the red horizontal line is the MSE/RMSE of the **baseline** regression model (without noise), majority of the noise added regression models perform better than the baseline model, with only one model performing poorly (that is: the k=4 neighbor noise added model) since it has a higher error metric as compared to the baseline model.

It is also important to note that the k=10 neighbor model has an error metric equivalent to the baseline model, since in the k=10, we are using all the data points to calculate the desired error term, and since an assumption of linear regression is that the sum of the error terms equals 0, then averaging 0 gives 0, 
which when added as noise is not different from the original baseline model.