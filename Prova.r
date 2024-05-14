#Read the time series in ar_data.csv
data <- read.csv("ar_data.csv", header = TRUE)

library(forecast)
library(tidyverse)

data$value <- as.numeric(data$value)

ts <- ts(data$value, frequency = 12)
ts

plot(ts)

library(gasmodel)

gas = gas(ts, distr = "t", scaling = "fisher_inv",  print_progress = TRUE, 
    par_static = c(FALSE, FALSE, FALSE), par_link = c(FALSE, FALSE, FALSE),
    optim_arguments = list(opts = list(algorithm = "NLOPT_LN_NELDERMEAD", xtol_rel =  1e-3, maxeval = 1000)))

print(gas)
plot(gas$fit$par_tv[,1])
plot(gas$fit$par_tv[,2])
plot(gas$fit$par_tv[,3])

#plot the original time series with the mean series gas$fit$par_tv[,1] and the 95% confidence interval (variance is gas$fit$par_tv[,2])
#The confidence interval should be represented as a light shaded area around the mean series

plot(gas$data$y, type = "l", col = "black", lwd = 2, ylab = "Value")
lines(gas$fit$par_tv[,1], col = "red", lwd = 2)
lines(gas$fit$par_tv[,1] + 1.96 * sqrt(gas$fit$par_tv[,2]), col = "red", lwd = 2, lty = 2)
lines(gas$fit$par_tv[,1] - 1.96 * sqrt(gas$fit$par_tv[,2]), col = "red", lwd = 2, lty = 2)
#color the area between the confidence interval






