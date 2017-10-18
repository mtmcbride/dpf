library(dpf)
nstates = 2
npart = 8
N = 35
n = 100
y = matrix(rnorm(n)+25,1)
a0 = matrix(rnorm(2*npart),2)
P0 = matrix(c(diag(.1,2)),4,npart)
w0 = runif(npart)
w0 = w0/sum(w0)
ct = array(0,c(3,2,1))
dt = array(0,c(2,2,1))
Zt = array(c(.1,.1,.1,.2,.3,.1),c(6,2,1))
Tt = array(c(diag(.9,2),diag(.8,2)),c(4,2,1))
Rt = array(c(diag(1,2)),c(4,2,1))
Qt = array(c(diag(1,2)),c(4,2,1))
GGt = array(c(diag(.1,3)),c(9,2,1))
currentStates = rbinom(npart,size = nstates-1,.5)
w = runif(npart)
w = w/sum(w)
currentStates[2] = 0
transProbs = matrix(c(.1,.9,.4,.6),2,byrow=TRUE)
lt = runif(n)
temposwitch = double(n)
temposwitch[floor(n/2):floor(3*n/4)] = 1
mus = c(60, 100, 2, 1)
sig2eps = 1
sig2eta = c(2, .1, 1)
transProbs = c(.8,.1,.5,.4)
toab <- function(x, a, b) x*(b-a) + a # maps [0,1] to [a,b]
logistic <- function(x) 1/(1+exp(-x)) # maps R to [0,1]

toOptimize <- function(pvec, lt, temposwitch, y, w0, Npart){
    sig2eps = exp(pvec[1])
    mus = pvec[2:5]
    sig2etas = exp(pvec[6:8])
    transprobs = logistic(pvec[9:12])
    transprobs[2] = toab(transprobs[2], 0, 1-transprobs[1])
    pmats = yupengMats(lt, temposwitch, sig2eps, mus, sig2etas, transprobs)
    S = beamSearch(pmats$a0, pmats$P0, w0, pmats$dt, pmats$ct, pmats$Tt, pmats$Zt,
                   pmats$Rt, pmats$Qt, pmats$GGt, y, pmats$transMat, Npart)
    if(S$LastStep < ncol(y)) return(Inf)
    best = S$paths[which.max(S$weights),]
    negllike = getloglike(pmats, best, y)
    return(negllike)
}

w0[2:4] = 0 #Michael's line
initParams = c(0, 25, 25, 1, 1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25)
badOnes = 0
for(i in 1:1000){
    testy = optim(initParams, fn = toOptimize, lt = lt, temposwitch = temposwitch, 
                  y = y, w0 = w0, Npart = npart, method = 'SANN')
    badOnes = badOnes + all(testy$par == initParams)
}
print(badOnes)
