dat <- read.csv("review_train_complete.csv")
# summary(dat)
# names(dat)
# dat$total = dat$useful+dat$cool+dat$funny
dat = na.omit(dat)
require(VGAM)
data = subset(dat, select = c(total_votes_x, stars_x, coleman_liau_index_x, automated_readability_index_x, dale_chall_readability_score_x, linsear_write_formula_x, gunning_fog_x, flesch_reading_ease_x, review_depth, fans, usr_avg_stars,usr_review_count,elite_count,usr_total_votes,usr_score,avg_nw_score,max_nw_score,friend_count,yelping_since,posted_days))

summary(zeroInflatedNB <- zeroinfl(total_votes_x ~ stars_x + coleman_liau_index_x + automated_readability_index_x + dale_chall_readability_score_x + linsear_write_formula_x + gunning_fog_x + flesch_reading_ease_x + review_depth + fans + usr_avg_stars  + elite_count  + usr_score + avg_nw_score + max_nw_score + friend_count  + posted_days, data = data, dist = "negbin", EM = TRUE))

data$yhat <- fitted(zeroInflatedNB)
(r <- with(data, cor(yhat, total_votes_x)))
r^2

mean((data$total_votes_x - data$yhat) ^ 2)

# dat1 <- read.csv("review_test_complete.csv")
# dat1 = na.omit(dat1)
dat1$yhat <- predict(zeroInflatedNB, dat1)
(r <- with(dat1, cor(yhat, total_votes_x)))
r^2

mean((dat1$total_votes_x - predict(zeroInflatedNB, dat1)) ^ 2)

R2<- 1-(zeroInflatedNB$deviance/zeroInflatedNB$null.deviance)
R2

mnull <- update(zeroInflatedNB, . ~ 1)
pchisq(2 * (logLik(zeroInflatedNB) - logLik(mnull)), df = 3, lower.tail = FALSE)


vuong(zeroInflatedNB, m)