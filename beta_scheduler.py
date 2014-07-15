import numpy as np
from matplotlib import pyplot as pp
import theano
import theano.tensor as T
import scipy as sp
from matplotlib import animation
from matplotlib.path import Path
import cPickle as cp
from sklearn import mixture
import copy

def whiten(x):
	mu=np.mean(x,axis=0)
	x=x-mu
	cov=np.cov(x.T)
	cov_inv=np.linalg.inv(cov)
	cov_inv_sqrt=sp.linalg.sqrtm(cov_inv)
	out=np.dot(x,cov_inv_sqrt)
	return out

### Making the swiss roll
nsamps=4000

data=np.random.rand(nsamps,2)*8.0+4.0
data=np.asarray([data[:,0]*np.cos(data[:,0]), data[:,0]*np.sin(data[:,0])])+np.random.randn(2,nsamps)*0.25
data=4.0*data.T

data=np.asarray(data, dtype='float32')
data0=whiten(data)*1.0

g0=mixture.GMM(n_components=200,n_iter=200)
g0.fit(data0)
gsamps=g0.sample(1000)
pp.scatter(gsamps[:,0],gsamps[:,1]); pp.show()

model_scores=g0.score(gsamps)
norm_scores=np.log(((2.0*np.pi)**(-1))*np.exp(-0.5*np.sum(gsamps**2,axis=1)))
initial_KL=np.mean(model_scores-norm_scores)
print initial_KL

nsteps=20

target_delta_KL=initial_KL/float(nsteps)

MIhist=[]
KLGhist=[]
KLhist=[]

betadict={}
for i in range(nsteps):
	
	beta=0.01
	means=data0*np.sqrt(1.0-beta) 
	data1=(means + np.random.randn(nsamps, 2)*np.sqrt(beta)).astype(np.float32)
	g1=mixture.GMM(n_components=200-8*i,n_iter=200)
	g1.fit(data1)
	g1samps=g1.sample(8000)
	g1_scores=g1.score(g1samps)
	g0_scores=g0.score(g1samps)
	KL=np.mean(g1_scores-g0_scores)
	KLhist.append(KL)
	#g0=copy.deepcopy(g1)
	g0=g1
	data0=data1
	
	#if i==0:
		#means=data0*np.sqrt(1.0-beta) 
		#data1=(means + np.random.randn(nsamps, 2)*np.sqrt(beta)).astype(np.float32)
		#diffs=data1.reshape((nsamps, 1, 2)) - means.reshape((1, nsamps, 2))
		#exp_terms=np.sum(diffs**2,axis=2)
		#cond_probs=((2.0*np.pi*beta)**(-1))*np.exp(-(1.0/(2*beta))*exp_terms)
		#p1_scores=np.log(np.sum(cond_probs,axis=1))
		#p0_scores=g.score(data1)
		#KL=np.mean(p1_scores-p0_scores)
		#old_means=means
		#old_beta=beta
		#data0=data1
		#KLhist.append(KL)
	#else:
		#means=data0*np.sqrt(1.0-beta) 
		#data1=(means + np.random.randn(nsamps, 2)*np.sqrt(beta)).astype(np.float32)
		#diffs1=data1.reshape((nsamps, 1, 2)) - means.reshape((1, nsamps, 2))
		#diffs0=data1.reshape((nsamps, 1, 2)) - old_means.reshape((1, nsamps, 2))
		#exp_terms0=np.sum(diffs0**2,axis=2)
		#exp_terms1=np.sum(diffs1**2,axis=2)
		#cond_probs1=((2.0*np.pi*beta)**(-1))*np.exp(-(1.0/(2*beta))*exp_terms1)
		#cond_probs0=((2.0*np.pi*old_beta)**(-1))*np.exp(-(1.0/(2*old_beta))*exp_terms0)
		#p1_scores=np.log(np.sum(cond_probs1,axis=1))
		#p0_scores=np.log(np.sum(cond_probs0,axis=1))
		#KL=np.mean(p1_scores-p0_scores)
		#old_means=means
		#old_beta=beta
		#data0=data1
		#KLhist.append(KL)
	
	#exp_terms=np.dot(data1,data1.T)-2.0*np.dot(data1,means.T)+np.dot(means,means.T)
	
	#For MI
	#cond_probs=((2.0*np.pi*beta)**(-1))*np.exp(-(1.0/(2*beta))*exp_terms)
	#term1=np.sum(np.log(np.diag(cond_probs)))
	#term2=np.sum(np.log((1.0/float(nsamps))*np.sum(cond_probs,axis=0)))
	#MI=(1.0/float(nsamps))*(term1-term2)
	#MIhist.append(MI)
	
	#For KL with a unit gaussian
	#exp_terms=np.sum(diffs**2,axis=2)
	#cond_probs=((2.0*np.pi*beta)**(-1))*np.exp(-(1.0/(2*beta))*exp_terms)
	#norm_probs=((2.0*np.pi)**(-1))*np.exp(-0.5*np.sum(data1**2,axis=1))
	#num_terms=np.mean(cond_probs,axis=1)
	#log_terms=np.log(num_terms/norm_probs)
	#KL=np.mean(log_terms)
	#KLGhist.append(KL)
	
	
	print i

KLhist=np.asarray(KLhist)


norm_scores=np.log(((2.0*np.pi)**(-1))*np.exp(-0.5*np.sum(g1samps**2,axis=1)))
end_KL=np.mean(g1_scores-norm_scores)
print np.sum(KLhist)
print end_KL
print np.sum(KLhist) + end_KL
pp.scatter(data1[:,0],data1[:,1])
pp.figure(2)
pp.plot(KLhist)
pp.show()
