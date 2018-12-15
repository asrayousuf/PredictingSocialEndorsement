
require(VGAM)
# data = subset(dat, select = c(total_votes_x, stars_x, coleman_liau_index_x, automated_readability_index_x, dale_chall_readability_score_x, linsear_write_formula_x, gunning_fog_x, flesch_reading_ease_x, review_depth, fans, usr_avg_stars,usr_review_count,elite_count,usr_total_votes,usr_score,avg_nw_score,max_nw_score,friend_count,yelping_since,posted_days))
require(pscl)
datBS <- read.csv("DataWithReadability_UserFeatures/train_BS.csv")
summary(dat)
names(datBS)
datBS = na.omit(datBS)
dataBS = subset(datBS, select = c(total_votes, stars, coleman_liau_index, automated_readability_index, dale_chall_readability_score, linsear_write_formula, gunning_fog, flesch_reading_ease, review_depth, fans, usr_avg_stars,usr_review_count,elite_count,usr_total_votes,usr_score,avg_nw_score,max_nw_score,friend_count,yelping_since,posted_days))

summary(tobitBS <- vglm(total_votes ~ stars + coleman_liau_index + automated_readability_index + dale_chall_readability_score + linsear_write_formula + gunning_fog + flesch_reading_ease + review_depth + fans + usr_avg_stars  + elite_count  + usr_score + avg_nw_score + max_nw_score + friend_count  + posted_days, tobit(Lower = 0), data = dataBS))
#summary(m <- vglm(apt ~ read + math + prog, tobit(Upper = 800), data = dat))
# PseudoR2(m)
ctable <- coef(summary(tobitBS))
pvals <- 2 * pt(abs(ctable[, "z value"]), df.residual(tobitBS), lower.tail = FALSE)
cbind(ctable, pvals)

dataBS$yhat <- fitted(tobitBS)[,1]
(r <- with(dataBS, cor(yhat, total_votes)))
r^2

meanY = mean(dataBS$total_votes)
denom = sum((dataBS$total_votes - meanY)^2)
numer = sum((dataBS$total_votes - dataBS$yhat)^2)
r2 = 1 - (numer/denom)
r2

mean((dataBS$total_votes - dataBS$yhat) ^ 2)

dataBSTest$yhat <- predict(tobitBS, dataBSTest)
(r <- with(dataBSTest, cor(yhat, total_votes)))
r^2

mean((dataBSTest$total_votes - predict(tobitBS, dataBSTest)) ^ 2)

R2<- 1-(tobitBS$deviance/tobitBS$null.deviance)
R2

mnull <- update(tobitBS, . ~ 1)
pchisq(2 * (logLik(tobitBS) - logLik(mnull)), df = 3, lower.tail = FALSE)
