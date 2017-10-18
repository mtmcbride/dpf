nstates = 2
npart = 8
N = 35
n = 100
y = matrix(1:3/10)
y = matrix(rnorm(3*n),3) #this one
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
w[2] = npart #not this one
w = w/sum(w)
currentStates[2] = 0
transProbs = matrix(c(.1,.9,.4,.6),2,byrow=TRUE)
out = beamSearch(a0, P0, w, dt, ct, Tt, Zt, Rt, Qt, GGt, y, transProbs, N)
HHt = HHcreate(Rt[,,1], Qt[,,1], nrow(a0), sqrt(nrow(Qt))) #Michael's code from here
ct = ct[,,1]
Qt = Qt[,,1]
Rt = Rt[,,1]
Tt = Tt[,,1]
Zt = Zt[,,1]
dt = dt[,,1]
GGt = GGt[,,1] #to here

out1 = dpf(currentStates, w, N, transProbs, a0, P0, dt, ct, Tt, Zt, HHt, GGt, y[,1]) #changed to only first column of y
ptm = proc.time()
#was beamSearch call was here
print(proc.time() - ptm)
lt = runif(n)
temposwitch = double(n)
temposwitch[floor(n/2):floor(3*n/4)] = 1
mus = c(60, 100, 2, 1)
sig2eps = 1
sig2eta = c(2, .1, 1)
transProbs = c(.8,.1,.5,.4)
ptm = proc.time()
out = yupengMats(lt, temposwitch, sig2eps, mus, sig2eta, transProbs)
test = beamSearch(a0,P0,w0, out$dt, out$ct, out$Tt, out$Zt, out$Rt, 
                 out$Qt, out$GGt, y, out$transMat, 50)
print(proc.time() - ptm)
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
testy = optim(c(0, 25, 25, 1, 1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25), fn = toOptimize, lt = lt, 
              temposwitch = temposwitch, y = y, w0 = w0, Npart = npart, method = 'SANN')

getPath <- function(pvec, lt, temposwitch, y, w0, Npart){
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
    ps = pathStuff(pmats, best, y)
    return(list(xpath = ps, spath = best))
}

getPmats <- function(pvec, lt, temposwitch, y, w0, Npart){
    sig2eps = exp(pvec[1])
    mus = pvec[2:5]
    sig2etas = exp(pvec[6:8])
    transprobs = logistic(pvec[9:12])
    transprobs[2] = toab(transprobs[2], 0, 1-transprobs[1])
    pmats = yupengMats(lt, temposwitch, sig2eps, mus, sig2etas, transprobs)
    return(pmats)
}

paths = getPath(testy$par, lt, temposwitch, y, w0, npart)
plot(as.vector(y) ~ c(1:n), pch = 19, 
     main = "the dots are values of y, the line is the first parameter of the continuous state",
     ylab = 'tempo', xlab = 'time')
lines(as.vector(paths$xpath) ~ c(1:n))
plot(as.vector(paths$spath) ~ c(1:n),
     main = 'discrete states over time',
     xlab = 'time', ylab = 'state', pch = 19)

########################################################
y[50:75] = matrix(rnorm(26))
testy = optim(c(0, 25, 0, 1, 1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25), fn = toOptimize, lt = lt, 
              temposwitch = temposwitch, y = y, w0 = w0, Npart = npart, method = 'SANN')
paths = getPath(testy$par, lt, temposwitch, y, w0, npart)
mats = getPmats(testy$par, lt, temposwitch, y, w0, npart)
plot(as.vector(y) ~ c(1:n), pch = 19, 
     main = "the dots are values of y, the line is the first parameter of the continuous state",
     ylab = 'tempo', xlab = 'time')
lines(as.vector(paths$xpath) ~ c(1:n))
plot(as.vector(paths$spath) ~ c(1:n),
     main = 'discrete states over time',
     xlab = 'time', ylab = 'state', pch = 19)

########################################################
n = 100
temposwitch = double(n)
temposwitch[floor(n/2):floor(3*n/4)] = 1
lt = rep(2, n)
sig2eps = 3
mus = c(4, 5, 6, 7)
sig2eta = c(8, 9, 10)
transProbs = c(.8,.1,.5,.4)
testmats = yupengMats(lt, temposwitch, sig2eps, mus, sig2eta, transProbs)
