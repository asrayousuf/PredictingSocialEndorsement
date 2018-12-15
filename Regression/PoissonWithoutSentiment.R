dat <- read.csv("review_train_complete.csv")
#summary(dat)

dat = na.omit(dat)
require(VGAM)
data = subset(dat, select = c(total_votes_x, stars_x, coleman_liau_index_x, automated_readability_index_x, dale_chall_readability_score_x, linsear_write_formula_x, gunning_fog_x, flesch_reading_ease_x, review_depth, fans, usr_avg_stars,usr_review_count,elite_count,usr_total_votes,usr_score,avg_nw_score,max_nw_score,friend_count,yelping_since,posted_days))

summary(poissonModelWSent <- glm(total_votes_x ~ stars_x + coleman_liau_index_x + automated_readability_index_x + dale_chall_readability_score_x + linsear_write_formula_x + gunning_fog_x + flesch_reading_ease_x + review_depth + fans + usr_avg_stars  + elite_count  + usr_score + avg_nw_score + max_nw_score + friend_count  + posted_days, family = "poisson", data = data))
#summary(m1 <- glm(num_awards ~ prog + math, family="poisson", data=p))

#summary(m <- vglm(apt ~ read + math + prog, tobit(Upper = 800), data = dat))
ctable <- coef(summary(poissonModelWSent))
pvals <- 2 * pt(abs(ctable[, "z value"]), df.residual(poissonModelWSent), lower.tail = FALSE)
cbind(ctable, pvals)

dat$yhat <- fitted(poissonModelWSent)
(r <- with(dat, cor(yhat, total_votes_x)))
r^2

mean((dat$total_votes_x - dat$yhat) ^ 2)

dat1 <- read.csv("review_test_complete.csv")
dat1 = na.omit(dat1)
dat1$yhat <- predict(poissonModelWSent, dat1)
(r <- with(dat1, cor(yhat, total_votes_x)))
r^2

mean((dat1$total_votes_x - predict(poissonModelWSent, dat1)) ^ 2)

R2<- 1-(poissonModelWSent$deviance/poissonModelWSent$null.deviance)
R2

mnull <- update(poissonModelWSent, . ~ 1)
pchisq(2 * (logLik(poissonModelWSent) - logLik(mnull)), df = 3, lower.tail = FALSE)

require(sandwich)
cov.poissonModelWSent <- vcovHC(poissonModelWSent, type="HC0")
std.err <- sqrt(diag(cov.poissonModelWSent))
r.est <- cbind(Estimate= coef(poissonModelWSent), "Robust SE" = std.err,
               "Pr(>|z|)" = 2 * pnorm(abs(coef(poissonModelWSent)/std.err), lower.tail=FALSE),
               LL = coef(poissonModelWSent) - 1.96 * std.err,
               UL = coef(poissonModelWSent) + 1.96 * std.err)

r.est

with(poissonModelWSent, cbind(res.deviance = deviance, df = df.residual,
                         p = pchisq(deviance, df.residual, lower.tail=FALSE)))

