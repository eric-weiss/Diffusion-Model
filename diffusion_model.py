import numpy as np
from matplotlib import pyplot as pp
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class diffusion_model():
	
	def __init__(self, nx, nb, nsteps, beta, nhid_mu, nhid_cov, ntgates=40):
		
		self.nx=nx
		self.nb=nb
		self.nsteps=nsteps
		self.beta=beta
		self.nhid_mu=nhid_mu
		self.nhid_cov=nhid_cov
		self.ntgates=ntgates
		
		self.kT=-np.log(0.5)*8.0*self.ntgates**2
		
		muW0=np.random.randn(nx, nhid_mu)*0.1
		muW1=np.random.randn(nhid_mu, ntgates*nx)*0.1
		mub0=np.zeros(nhid_mu)
		mub1=np.zeros(nx)
		
		covW0=np.random.randn(nx, nhid_cov)*0.1
		covW1=np.random.randn(nhid_cov, ntgates*nx)*0.1
		covb0=np.zeros(nhid_cov)
		covb1=np.zeros(nx)
		
		
		#muWT=np.random.randn(1,ntgates)*0.0
		#covWT=np.random.randn(1,ntgates)*0.0
		#mubT=np.zeros(ntgates)
		#covbT=np.zeros(ntgates)
		
		#muWT[0,1]=16.0; muWT[0,2]=32.0
		#mubT[0]=8.0; mubT[2]=-24.0
		#covWT[0,1]=16.0; covWT[0,2]=32.0
		#covbT[0]=8.0; covbT[2]=-24.0
		
		self.muW0=theano.shared(np.asarray(muW0,dtype='float32'))
		self.muW1=theano.shared(np.asarray(muW1,dtype='float32'))
		self.mub0=theano.shared(np.asarray(mub0,dtype='float32'))
		self.mub1=theano.shared(np.asarray(mub1,dtype='float32'))
		self.covW0=theano.shared(np.asarray(covW0,dtype='float32'))
		self.covW1=theano.shared(np.asarray(covW1,dtype='float32'))
		self.covb0=theano.shared(np.asarray(covb0,dtype='float32'))
		self.covb1=theano.shared(np.asarray(covb1,dtype='float32'))
		
		#self.muWT=theano.shared(np.asarray(muWT*2.0,dtype='float32'))
		#self.covWT=theano.shared(np.asarray(covWT*2.0,dtype='float32'))
		#self.mubT=theano.shared(np.asarray(mubT,dtype='float32'))
		#self.covbT=theano.shared(np.asarray(covbT,dtype='float32'))
		
		self.theano_rng = RandomStreams()
		
		self.params=[self.muW0, self.muW1, self.mub0, self.mub1,
					self.covW0, self.covW1, self.covb0, self.covb1]
					#self.muWT, self.covWT, self.mubT, self.covbT]
		
		self.true_params=[]
		self.momentums=[]
		
		for param in self.params:
			val=param.get_value()
			self.true_params.append(theano.shared(val))
			self.momentums.append(theano.shared(val*0.0))
		
	
	
	def compute_f_mu(self, x, t):
		
		h=T.nnet.sigmoid(T.dot(x,self.muW0)+self.mub0) #nt by nb by nhidmu
		z=T.dot(h,self.muW1)
		z=T.reshape(z,(t.shape[0],t.shape[1],self.ntgates,self.nx))+self.mub1 #nt by nb by ntgates by nx
		#z=z+T.reshape(x,(t.shape[0],t.shape[1],1,self.nx))
		
		tpoints=T.cast(T.arange(self.ntgates),'float32')/T.cast(self.ntgates-1,'float32')
		tpoints=T.reshape(tpoints, (1,1,self.ntgates))
		#tgating=T.exp(T.dot(t,self.muWT)+self.mubT) #nt by nb by ntgates
		tgating=T.exp(-self.kT*(tpoints-t)**2)
		tgating=tgating/T.reshape(T.sum(tgating, axis=2),(t.shape[0], t.shape[1], 1))
		tgating=T.reshape(tgating,(t.shape[0],t.shape[1],self.ntgates,1))
		
		mult=z*tgating
		
		out=T.sum(mult,axis=2)
		
		out=out+x
		
		return T.cast(out,'float32')
	
	
	def compute_f_cov(self, x, t):
		
		h=T.nnet.sigmoid(T.dot(x,self.covW0)+self.covb0) #nt by nb by nhidmu
		z=T.dot(h,self.covW1)
		z=T.reshape(z,(t.shape[0],t.shape[1],self.ntgates,self.nx))+self.covb1 #nt by nb by ntgates by 1
		z=T.exp(z)
		
		tpoints=T.cast(T.arange(self.ntgates),'float32')/T.cast(self.ntgates-1,'float32')
		tpoints=T.reshape(tpoints, (1,1,self.ntgates))
		#tgating=T.exp(T.dot(t,self.covWT)+self.covbT) #nt by nb by ntgates
		tgating=T.exp(-self.kT*(tpoints-t)**2)
		tgating=tgating/T.reshape(T.sum(tgating, axis=2),(t.shape[0], t.shape[1], 1))
		tgating=T.reshape(tgating,(t.shape[0],t.shape[1],self.ntgates,1))
		
		mult=z*tgating
		
		out=T.sum(mult,axis=2)
		
		return T.cast(out,'float32')
	
	
	def forward_step(self, x, t):
		
		samps=self.theano_rng.normal(size=x.shape)*T.sqrt(self.beta+t)
		means=x*T.sqrt(1.0-(self.beta+t))
		return T.cast(means+samps,'float32'), T.cast(t+0.05/200.0,'float32')
	
	
	def compute_forward_trajectory(self, x0):
		
		[x_seq, ts], updates=theano.scan(fn=self.forward_step,
										outputs_info=[x0, 0.0],
										n_steps=self.nsteps)
		return x_seq, updates
	
	
	def loss(self, x_seq, t):
		
		f_mu=self.compute_f_mu(x_seq,t)
		f_cov=self.compute_f_cov(x_seq,t)
		#f_cov=T.extra_ops.repeat(f_cov,self.nx,axis=2)
		diffs=(f_mu[1:]-x_seq[:-1])**2
		gaussian_terms=T.sum(diffs*(1.0/f_cov[1:]),axis=2)
		det_terms=T.sum(T.log(f_cov[1:]),axis=2)
		
		return gaussian_terms+det_terms
	
	
	def get_loss(self, x_seq):
		t=T.cast(T.arange(self.nsteps),'float32')/T.cast(self.nsteps,'float32')
		t=T.reshape(t,(self.nsteps,1,1))
		t=T.extra_ops.repeat(t,self.nb,axis=1)
		objective=self.loss(x_seq,t)
		return T.mean(T.mean(objective))
	
	
	def train_step_nosample(self, x_seq, lrate):
		
		t0=T.cast(T.arange(self.nsteps),'float32')/T.cast(self.nsteps,'float32')
		t=T.reshape(t0,(self.nsteps,1,1))
		t=T.extra_ops.repeat(t,self.nb,axis=1)
		loss_terms=self.loss(x_seq,t)
		objective=T.mean(T.mean(loss_terms))
		updates={}
		gparams=T.grad(objective, self.params, consider_constant=[x_seq,t])
		
		mu=0.98
		
		## constructs the update dictionary
		for gparam, param, tparam, momentum in zip(gparams, self.params, self.true_params, self.momentums):
			
			mult=1.0
			#if param==self.muWT or param==self.mubT or param==self.covWT or param==self.covbT:
			#	mult=0.0
			new_momentum=mu*momentum-lrate*gparam*mult
			new_tparam=tparam+new_momentum
			new_param=new_tparam+mu*new_momentum
			updates[param] = T.cast(new_param,'float32')
			#updates[param] = T.cast(param-lrate*gparam*mult,'float32')
			updates[tparam] = T.cast(new_tparam,'float32')
			updates[momentum] = T.cast(new_momentum,'float32')
		
		
		extra=self.compute_f_cov(x_seq,t)
		
		return objective, extra[:,0], updates
	
	
	def reverse_step(self, x, t, nsamps):
		
		f_mu=self.compute_f_mu(x,t)
		f_cov=self.compute_f_cov(x,t)
		#f_cov=T.extra_ops.repeat(f_cov,self.nx,axis=2)
		samps=self.theano_rng.normal(size=(1,nsamps, self.nx))
		samps=samps*T.sqrt(f_cov)+f_mu
		return samps,T.cast(t-1.0/self.nsteps,'float32')
	
	
	def get_samps(self, nsamps):
		
		t=1.0
		t=T.reshape(t,(1,1,1))
		t=T.extra_ops.repeat(t,nsamps,axis=1)
		t=T.cast(t,'float32')
		x0=self.theano_rng.normal(size=(nsamps, self.nx))
		x0=T.reshape(x0,(1,nsamps,self.nx))
		[samphist, ts], updates=theano.scan(fn=self.reverse_step,
										outputs_info=[x0,t],
										non_sequences=nsamps,
										n_steps=self.nsteps+1)
		return samphist[:,0,:,:], ts[:,0], updates
		
		
	def get_tgating(self):
		t0=T.cast(T.arange(self.nsteps),'float32')/T.cast(self.nsteps,'float32')
		t=T.reshape(t0,(self.nsteps,1,1))
		t=T.extra_ops.repeat(t,1,axis=1)
		tpoints=T.cast(T.arange(self.ntgates),'float32')/T.cast(self.ntgates-1,'float32')
		tpoints=T.reshape(tpoints, (1,1,self.ntgates))
		#tgating=T.exp(T.dot(t,self.muWT)+self.mubT) #nt by nb by ntgates
		tgating=T.exp(-self.kT*(tpoints-t)**2)
		tgating=tgating/T.reshape(T.sum(tgating, axis=2),(t.shape[0], t.shape[1], 1))
		tgating=T.reshape(tgating,(t.shape[0],t.shape[1],self.ntgates,1))
		return tgating
		
		
		
		
