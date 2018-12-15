# summary(m1 <- glm.nb(daysabs ~ math + prog, data = dat))

dat <- read.csv("review_train_complete.csv")
# summary(dat)
# names(dat)
# dat$total = dat$useful+dat$cool+dat$funny
require(MASS)

dat = na.omit(dat)
require(VGAM)
data1 = subset(dat, select = c(total_votes_x, stars_x, coleman_liau_index_x, automated_readability_index_x, dale_chall_readability_score_x, linsear_write_formula_x, gunning_fog_x, flesch_reading_ease_x, review_depth, fans, usr_avg_stars,usr_review_count,elite_count,usr_total_votes,usr_score,avg_nw_score,max_nw_score,friend_count,yelping_since,posted_days))

summary(nb <- glm.nb(total_votes_x ~ stars_x + coleman_liau_index_x + automated_readability_index_x + dale_chall_readability_score_x + linsear_write_formula_x + gunning_fog_x + flesch_reading_ease_x + review_depth + fans + usr_avg_stars  + elite_count  + usr_score + avg_nw_score + max_nw_score + friend_count  + posted_days, data = data1))
#summary(m <- vglm(apt ~ read + math + prog, tobit(Upper = 800), data = dat))
# PseudoR2(m)
ctable <- coef(summary(nb))
pvals <- 2 * pt(abs(ctable[, "z value"]), df.residual(nb), lower.tail = FALSE)
cbind(ctable, pvals)

data1$yhat <- fitted(nb)
(r <- with(data1, cor(yhat, total_votes_x)))
r^2

mean((data1$total_votes_x - data1$yhat) ^ 2)

dat1 <- read.csv("review_test_complete.csv")
dat1 = na.omit(dat1)
dat1$yhat <- predict(nb, dat1)
(r <- with(dat1, cor(yhat, total_votes_x)))
r^2

mean((dat1$total_votes_x - predict(nb, dat1)) ^ 2)

R2<- 1-(nb$deviance/nb$null.deviance)
R2

mnull <- update(nb, . ~ 1)
pchisq(2 * (logLik(nb) - logLik(mnull)), df = 3, lower.tail = FALSE)
