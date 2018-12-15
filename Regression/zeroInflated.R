require(pscl)
# summary(m1 <- zeroinfl(count ~ child + camper | persons, data = zinb))
summary(zeroInflatedPoisson <- zeroinfl(total_votes_x ~ stars_x + coleman_liau_index_x + automated_readability_index_x + dale_chall_readability_score_x + linsear_write_formula_x + gunning_fog_x + flesch_reading_ease_x + review_depth + fans + usr_avg_stars  + elite_count  + usr_score + avg_nw_score + max_nw_score + friend_count  + posted_days, data = data))

data$yhat <- fitted(zeroInflatedPoisson)
(r <- with(data, cor(yhat, total_votes_x)))
r^2

mean((data$total_votes_x - data$yhat) ^ 2)

# dat1 <- read.csv("review_test_complete.csv")
# dat1 = na.omit(dat1)
dat1$yhat <- predict(zeroInflatedPoisson, dat1)
(r <- with(dat1, cor(yhat, total_votes_x)))
r^2

mean((dat1$total_votes_x - predict(zeroInflatedPoisson, dat1)) ^ 2)

R2<- 1-(zeroInflatedPoisson$deviance/zeroInflatedPoisson$null.deviance)
R2

mnull <- update(zeroInflatedPoisson, . ~ 1)
pchisq(2 * (logLik(zeroInflatedPoisson) - logLik(mnull)), df = 3, lower.tail = FALSE)


vuong(zeroInflatedPoisson, zeroInflatedNB)