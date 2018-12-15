require(pscl)
datBS <- read.csv("DataWithReadability_UserFeatures/train_BS.csv")
summary(dat)
names(datBS)
datBS = na.omit(datBS)
dataBS = subset(datBS, select = c(total_votes, stars, coleman_liau_index, automated_readability_index, dale_chall_readability_score, linsear_write_formula, gunning_fog, flesch_reading_ease, review_depth, fans, usr_avg_stars,usr_review_count,elite_count,usr_total_votes,usr_score,avg_nw_score,max_nw_score,friend_count,yelping_since,posted_days))
# summary(m1 <- zeroinfl(count ~ child + camper | persons, data = zinb))
summary(zeroInfPoissonBS <- zeroinfl(total_votes ~ stars + coleman_liau_index + automated_readability_index + dale_chall_readability_score + linsear_write_formula + gunning_fog + flesch_reading_ease + review_depth + fans + usr_avg_stars  + elite_count  + usr_score + avg_nw_score + max_nw_score + friend_count  + posted_days, data = dataBS))

dataBS$yhat <- fitted(zeroInfPoissonBS)
(r <- with(dataBS, cor(yhat, total_votes)))
r^2

mean((dataBS$total_votes - dataBS$yhat) ^ 2)

dataBSTest <- read.csv("DataWithReadability_UserFeatures/test_BS.csv")
names(dataBS)
dataBSTest$yhat <- predict(zeroInfPoissonBS, dataBSTest)
(r <- with(dataBSTest, cor(yhat, total_votes)))
r^2
mean((dataBSTest$total_votes - predict(zeroInfPoissonBS, dataBSTest)) ^ 2)

vuong(poissonModel, m)

