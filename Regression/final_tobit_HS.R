dat <- read.csv("finalData/train_reviews_HS.csv")
# dat <- read.csv("imp_train_restaurants.csv")
# summary(dat)
names(dat)
# dat$total = dat$useful+dat$cool+dat$funny
dat$review_length =  apply(dat, 1, FUN = function(x) nchar(x['text']))

dat = na.omit(dat)
require(VGAM)
# data = subset(dat, select = c(total_votes, stars,  review_length, fans, usr_avg_stars,usr_review_count,elite_count,usr_total_votes,usr_score,avg_nw_score,max_nw_score,friend_count,yelping_since,posted_days, sentiment))
# data1 = data

# library(dplyr)
# df1 <- filter(data1, total_votes_x>0)
# data = subset(dat, select = c(total_votes_x, stars_x, coleman_liau_index_x, automated_readability_index_x, dale_chall_readability_score_x, linsear_write_formula_x, gunning_fog_x, flesch_reading_ease_x, review_length))
# nrow(df1)
# nrow(data1)
# 
# data = df1
data = dat[ , !(names(dat) %in% c('review_id', 'business_id', 'user_id', 'date', 'text', 'useful', 'cool', 'funny', 'usr_review_count','usr_total_votes', 'yelping_since'))]
names(data)
# summary(hsTobit <- vglm(total_votes ~ stars + coleman_liau_index + automated_readability_index + dale_chall_readability_score + linsear_write_formula + gunning_fog + flesch_reading_ease + review_length + fans + usr_avg_stars  + elite_count  + usr_score + avg_nw_score + max_nw_score + friend_count  + posted_days+ sentiment, family = tobit(Lower = 0), data=data))
#summary(m <- vglm(apt ~ read + math + prog, tobit(Upper = 800), data = dat))
# PseudoR2(m)
# ctable <- coef(summary(hsTobit))
# pvals <- 2 * pt(abs(ctable[, "z value"]), df.residual(hsTobit), lower.tail = FALSE)
# cbind(ctable, pvals)
summary(hsTobit <- vglm(total_votes ~ stars + readability_standard + review_length + fans + usr_avg_stars  + elite_count  + usr_score + avg_nw_score + max_nw_score + friend_count  + posted_days+ sentiment + Topic..0 +Topic..1 +Topic..2 +Topic..3 +Topic..4 +Topic..5 +Topic..6 +Topic..7 +Topic..8 +Topic..9 +Topic..10 +Topic..11 +Topic..12 +Topic..13 +Topic..14 +Topic..15 +Topic..16 +Topic..17 +Topic..18 +Topic..19 +Topic..20 +Topic..21 +Topic..22 +Topic..23 +Topic..24 +Topic..25 +Topic..26 +Topic..27 +Topic..28 +Topic..29 +Topic..30 +Topic..31 +Topic..32 +Topic..33 +Topic..34 +Topic..35 +Topic..36 +Topic..37 +Topic..38 +Topic..39 +Topic..40 +Topic..41 +Topic..42 +Topic..43 +Topic..44 +Topic..45 +Topic..46 +Topic..47 +Topic..48 +Topic..49 +Topic..50 +Topic..51 +Topic..52 +Topic..53 +Topic..54 +Topic..55 +Topic..56 +Topic..57 +Topic..58 +Topic..59 +Topic..60 +Topic..61 +Topic..62 +Topic..63 +Topic..64 +Topic..65 +Topic..66 +Topic..67 +Topic..68 +Topic..69 +Topic..70 +Topic..71 +Topic..72 +Topic..73 +Topic..74 +Topic..75 +Topic..76 +Topic..77 +Topic..78 +Topic..79 +Topic..80 +Topic..81 +Topic..82 +Topic..83 +Topic..84 +Topic..85 +Topic..86 +Topic..87 +Topic..88 +Topic..89 +Topic..90 +Topic..91+Topic..92+Topic..93+Topic..94+Topic..95+Topic..96+Topic..97+Topic..98, family = tobit(Lower = 0), data=data))
# summary(bshsTobit100 <- vglm(total_votes ~ stars + readability_standard + review_length + fans + usr_avg_stars  + elite_count  + usr_score + avg_nw_score + max_nw_score + friend_count  + posted_days+ sentiment + Topic..0 +Topic..1 +Topic..2 +Topic..3 +Topic..4 +Topic..5 +Topic..6 +Topic..7 +Topic..8 +Topic..9 +Topic..10 +Topic..11 +Topic..12 +Topic..13 +Topic..14 +Topic..15 +Topic..16 +Topic..17 +Topic..18 +Topic..19 +Topic..20 +Topic..21 +Topic..22 +Topic..23 +Topic..24 +Topic..25 +Topic..26 +Topic..27 +Topic..28 +Topic..29 +Topic..30 +Topic..31 +Topic..32 +Topic..33 +Topic..34 +Topic..35 +Topic..36 +Topic..37 +Topic..38 +Topic..39 +Topic..40 +Topic..41 +Topic..42 +Topic..43 +Topic..44 +Topic..45 +Topic..46 +Topic..47 +Topic..48 +Topic..49 +Topic..50 +Topic..51 +Topic..52 +Topic..53 +Topic..54 +Topic..55 +Topic..56 +Topic..57 +Topic..58 +Topic..59 +Topic..60 +Topic..61 +Topic..62 +Topic..63 +Topic..64 +Topic..65 +Topic..66 +Topic..67 +Topic..68 +Topic..69 +Topic..70 +Topic..71 +Topic..72 +Topic..73 +Topic..74 +Topic..75 +Topic..76 +Topic..77 +Topic..78 +Topic..79 +Topic..80 +Topic..81 +Topic..82 +Topic..83 +Topic..84 +Topic..85 +Topic..86 +Topic..87 +Topic..88 +Topic..89 +Topic..90 +Topic..91+Topic..92+Topic..93+Topic..94+Topic..95+Topic..96+Topic..97+Topic..98 +Topic..99, family = tobit(Lower = 0), data=data))

# summary(hsTobit <- vglm(total_votes ~ . , tobit(Lower = 0), data=data))
dat$yhat <- fitted(hsTobit)[,1]
(r <- with(dat, cor(yhat, total_votes)))
r^2

sqrt(mean((dat$total_votes - dat$yhat) ^ 2))
mean((dat$total_votes - dat$yhat) ^ 2)
dat1S <- read.csv("finalData/test_reviews_HS.csv")
dat1S = na.omit(dat1S)
dat1S$review_length =  apply(dat1S, 1, FUN = function(x) nchar(x['text']))

dat1S$yhat <- predict(hsTobit, dat1S)
(r <- with(dat1S, cor(yhat, total_votes)))
r^2

sqrt(mean((dat1S$total_votes - predict(hsTobit, dat1S)) ^ 2))
mean((dat1S$total_votes - predict(hsTobit, dat1S)) ^ 2)
# R2<- 1-(hsTobit$deviance/hsTobit$null.deviance)
# R2
# 
mnull <- update(hsTobit, . ~ 1)
pchisq(2 * (logLik(hsTobit) - logLik(mnull)), df = 3, lower.tail = FALSE)

# dat$yhat <- fitted(hsTobit)[,1]
# dat$rr <- resid(hsTobit, type = "response")
# dat$rp <- resid(hsTobit, type = "pearson")[,1]
# 
# par(mfcol = c(2, 3))
# 
# with(dat, {
#   plot(yhat, rr, main = "Fitted vs Residuals")
#   qqnorm(rr)
#   plot(yhat, rp, main = "Fitted vs Pearson Residuals")
#   qqnorm(rp)
#   plot(total_votes_x, rp, main = "Actual vs Pearson Residuals")
#   plot(total_votes_x, yhat, main = "Actual vs Fitted")
# })

with(dat, {plot(total_votes, yhat, main = "Actual vs Fitted")})