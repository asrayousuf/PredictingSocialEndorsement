# summary(m1 <- glm.nbBS(daysabs ~ math + prog, data = dat))

# dat <- read.csv("review_train_complete.csv")
# # summary(dat)
# # names(dat)
# # dat$total = dat$useful+dat$cool+dat$funny
require(MASS)

# dat = na.omit(dat)
# require(VGAM)
# dataBS = subset(dat, select = c(total_votes_x, stars_x, coleman_liau_index_x, automated_readability_index_x, dale_chall_readability_score_x, linsear_write_formula_x, gunning_fog_x, flesch_reading_ease_x, review_depth, fans, usr_avg_stars,usr_review_count,elite_count,usr_total_votes,usr_score,avg_nw_score,max_nw_score,friend_count,yelping_since,posted_days))

summary(nbBS <- glm.nb(total_votes ~ stars + coleman_liau_index + automated_readability_index + dale_chall_readability_score + linsear_write_formula + gunning_fog + flesch_reading_ease + review_depth + fans + usr_avg_stars  + elite_count  + usr_score + avg_nw_score + max_nw_score + friend_count  + posted_days, data = dataBS))
#summary(m <- vglm(apt ~ read + math + prog, tobit(Upper = 800), data = dat))
# PseudoR2(m)
ctable <- coef(summary(nbBS))
pvals <- 2 * pt(abs(ctable[, "z value"]), df.residual(nbBS), lower.tail = FALSE)
cbind(ctable, pvals)

dataBS$yhat <- fitted(nbBS)
(r <- with(dataBS, cor(yhat, total_votes)))
r^2

mean((dataBS$total_votes - dataBS$yhat) ^ 2)

dataBSTest$yhat <- predict(nbBS, dataBSTest)
(r <- with(dataBSTest, cor(yhat, total_votes)))
r^2

mean((dataBSTest$total_votes - predict(nbBS, dataBSTest)) ^ 2)

R2<- 1-(nbBS$deviance/nbBS$null.deviance)
R2

mnull <- update(nbBS, . ~ 1)
pchisq(2 * (logLik(nbBS) - logLik(mnull)), df = 3, lower.tail = FALSE)
