#--------------------------preparation-----------------------------------------------
library(xgboost)
setwd("C://Users//tw2567//Downloads")
train=read.csv("xgboost train.csv",as.is=TRUE)
test=read.csv("xgboost test.csv",as.is=TRUE)
xtrain=train[,2:12]
ytrain=train[,13]
xtest=test[,2:12]
dtrain = xgb.DMatrix(as.matrix(xtrain), label=ytrain)
dtest  = xgb.DMatrix(as.matrix(xtest))

#-------------------------tune parameters----------------------------------------------
params=list(
  eta = 0.1,
  objective = 'reg:linear',
  max_depth= 6,
  min_child_weight= 3,
  gamma=0.2,
  subsample=0.8
)
res = xgb.cv(params,
             dtrain,
             nrounds=10000,              #--- Will stop before (early_stopping_rounds)
             nfold=10,
             early_stopping_rounds=100,
             print_every_n=100,
             verbose=2,
             eval_metric = "rmse",
             maximize=FALSE)

#--------------------------------train-----------------s-------------------
params=list(
  eta = 0.1,
  objective = 'reg:linear',
  max_depth= 8,
  min_child_weight= 5,
  subsample=0.8
)
gbdt = xgb.train(params, dtrain, 5000)

#-------------------------predict and save the result-------------------------------
pred5000 = predict(gbdt,dtest)
pred5000=exp(pred5000)

preds_format=read.csv("customer_preds.csv",as.is = TRUE)
preds_number=data.frame('customer_id'=1:20000,pred5000)

for (i in 1:20000){
  id=preds_format$customer_id[i]
  preds_format$three_month_pred[i]=preds_number$pred5000[preds_number$customer_id==id]
}
write.csv(preds_format,"result_xgboost_5000_rounds.csv",row.names = FALSE)

#---------------------get importance matrix--------------------------------

names=names(train)[2:12]
importance_matrix <- xgb.importance(names, model = gbdt)

write.csv(importance_matrix,"xgboost importance matrix.csv",row.names = FALSE)
