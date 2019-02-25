rm(list = ls())
library(lsa)
require(xlsx)
require(recommenderlab)  #incele. 
#https://cran.r-project.org/web/packages/recommenderlab/vignettes/recommenderlab.pdf
critics <- read.xlsx("C:/Users/ecetp/Downloads/MEF/BDA 523 - Marketing Analytics/Week4/W01 - critics.xlsx",sheetIndex = 1)
#cosine similarity calculation
x  = critics[,2:ncol(critics)]
x[is.na(x)] = 0
user_sim = cosine(as.matrix(t(x))) #user similarity
userNo = 1


#1.method from Ozgür Hoca:
rec_itm_for_user = function(userNo) 
{ 
  weight_mat = user_sim[,userNo]*critics[,2:ncol(critics)] 
  #calculate column wise sum 
  col_sums= list()
  rat_user = critics[userNo,2:ncol(critics)]
  x=1 
  tot = list()
  z=1
  for(i in 1:ncol(rat_user)){ 
    if(is.na(rat_user[1,i])) 
    { 
      col_sums[x] = sum(weight_mat[,i],na.rm=TRUE)
      x=x+1
      temp = as.data.frame(weight_mat[,i])
      sum_temp=0
      for(j in 1:nrow(temp))
      { if(!is.na(temp[j,1]))
      {
        sum_temp = sum_temp+user_sim[j,userNo]
      }
      } 
      tot[z] = sum_temp 
      z=z+1 
    }
  }
  z=NULL
  z=1
  for(i in 1:ncol(rat_user)){ 
    if(is.na(rat_user[1,i]))
    {
      rat_user[1,i] = col_sums[[z]]/tot[[z]] 
      z=z+1 
    }
  } 
  return(rat_user)
}

result <- as.data.frame(rec_itm_for_user(2))

#######2.method: 

critics_m <- as(critics,"matrix")

ratings<- as(critics_m,"realRatingMatrix") #recomLab

Recom_model <- Recommender(data = ratings,method = "UBCF") # UBCF = User based, IBCF=Item Based 

recommended.items.1 <- predict(Recom_model, ratings[1,], n=5)  # çok film çikabilir; Top_N mantigi ile seçmis bu sebeple. Top 5 i aliyor.

### With Params

Recom_model=Recommender(ratings,method="UBCF", 
                      param=list(normalize = "Z-score",method="Cosine",nn=5, minRating=1))

recommended.items.1 <- predict(Recom_model, ratings[1,], n=5) # ratings matrixindeki 1.user için prediction yapiyorum.
#recommended.items.1 <- predict(Recom_model, ratings[1,], n=5) # tüm kullanicilar için top 5'i çeker.


as(recommended.items.1, "list")


