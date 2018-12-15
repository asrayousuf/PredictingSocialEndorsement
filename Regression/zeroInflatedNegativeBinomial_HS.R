require(pscl)
datHS <- read.csv("DataWithReadability_UserFeatures/train_HS.csv")
summary(dat)
names(datHS)
datHS = na.omit(datHS)
dataHS = subset(datHS, select = c(total_votes, stars, coleman_liau_index, automated_readability_index, dale_chall_readability_score, linsear_write_formula, gunning_fog, flesch_reading_ease, review_depth, fans, usr_avg_stars,usr_review_count,elite_count,usr_total_votes,usr_score,avg_nw_score,max_nw_score,friend_count,yelping_since,posted_days))
# summary(m1 <- zeroinfl(count ~ child + camper | persons, data = zinb))
summary(zeroInfNegBinHS <- zeroinfl(total_votes ~ stars + coleman_liau_index + automated_readability_index + dale_chall_readability_score + linsear_write_formula + gunning_fog + flesch_reading_ease + review_depth + fans + usr_avg_stars  + elite_count  + usr_score + avg_nw_score + max_nw_score + friend_count  + posted_days, data = dataHS))

dataHS$yhat <- fitted(zeroInfNegBinHS)
(r <- with(dataHS, cor(yhat, total_votes)))
r^2

mean((dataHS$total_votes - dataHS$yhat) ^ 2)

dataHSTest <- read.csv("DataWithReadability_UserFeatures/test_HS.csv")
names(dataBS)
dataHSTest$yhat <- predict(zeroInfNegBinHS, dataHSTest)
(r <- with(dataHSTest, cor(yhat, total_votes)))
r^2

vuong(poissonModel, m)

mean((dataHSTest$total_votes - predict(zeroInfNegBinHS, dataHSTest)) ^ 2)

pR2 = 1 - zeroInfNegBinHS$deviance / zeroInfNegBinHS$null.deviance 
1- logLik(mod)/logLik(mod_null)
require(BaylorEdPsych)
PseudoR2(zeroInfNegBinHS)
summary(zeroInfNegBinHS)