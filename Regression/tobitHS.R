require(pscl)
datHS <- read.csv("DataWithReadability_UserFeatures/train_HS.csv")
summary(dat)
names(datHS)
datHS = na.omit(datHS)
dataHS = subset(datHS, select = c(total_votes, stars, coleman_liau_index, automated_readability_index, dale_chall_readability_score, linsear_write_formula, gunning_fog, flesch_reading_ease, review_depth, fans, usr_avg_stars,usr_review_count,elite_count,usr_total_votes,usr_score,avg_nw_score,max_nw_score,friend_count,yelping_since,posted_days))

dataHSTest <- read.csv("DataWithReadability_UserFeatures/test_HS.csv")

summary(tobitHS <- vglm(total_votes ~ stars + coleman_liau_index + automated_readability_index + dale_chall_readability_score + linsear_write_formula + gunning_fog + flesch_reading_ease + review_depth + fans + usr_avg_stars  + elite_count  + usr_score + avg_nw_score + max_nw_score + friend_count  + posted_days, tobit(Lower = 0), data = dataHS))
#summary(m <- vglm(apt ~ read + math + prog, tobit(Upper = 800), data = dat))
# PseudoR2(m)
ctable <- coef(summary(tobitHS))
pvals <- 2 * pt(abs(ctable[, "z value"]), df.residual(tobitHS), lower.tail = FALSE)
cbind(ctable, pvals)

dataHS$yhat <- fitted(tobitHS)[,1]
(r <- with(dataHS, cor(yhat, total_votes)))
r^2

mean((dataHS$total_votes - dataHS$yhat) ^ 2)

dataHSTest$yhat <- predict(tobitHS, dataHSTest)
(r <- with(dataHSTest, cor(yhat, total_votes)))
r^2

mean((dataHSTest$total_votes - predict(tobitHS, dataHSTest)) ^ 2)

R2<- 1-(tobitHS$deviance/tobitHS$null.deviance)
R2

mnull <- update(tobitHS, . ~ 1)
pchisq(2 * (logLik(tobitHS) - logLik(mnull)), df = 3, lower.tail = FALSE)
