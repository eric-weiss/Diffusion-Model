import numpy as np
from matplotlib import pyplot as pp
import theano
import theano.tensor as T
import scipy as sp
from matplotlib import animation
from matplotlib.path import Path

import sys
sys.path.append('/home/float/Desktop/Sum-of-Functions-Optimizer/')
from sfo import SFO

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


nx=2
nsamps=4000
#n_subfuncs=int(np.round(np.sqrt(nsamps)/10.0))
#batchsize=int(np.round(10.0*np.sqrt(nsamps)))
n_subfuncs=100
batchsize=int(nsamps/n_subfuncs)
nsteps=100
beta=1. - np.exp(np.log(0.1)/float(nsteps))
nhid_mu=96
nhid_cov=2
nout_mu=96
nout_cov=2
ntgates=10

save_forward_animation=False

kT=-np.log(0.5)*8.0*ntgates**2

muW0=(np.random.randn(nx, nhid_mu)*1.1).astype(np.float32)
muW1=(np.random.randn(nhid_mu, nout_mu)*1.1).astype(np.float32)
muW2=(np.random.randn(nout_mu, ntgates*nx)*1.1).astype(np.float32)
mub0=np.zeros(nhid_mu).astype(np.float32)
mub1=np.zeros(nout_mu).astype(np.float32)
mub2=np.zeros(nx).astype(np.float32)

covW0=(np.random.randn(nx, nhid_cov)*1.1).astype(np.float32)
covW1=(np.random.randn(nhid_cov, nout_cov)*1.1).astype(np.float32)
covW2=(np.zeros((nout_cov, ntgates*nx))).astype(np.float32)
covb0=np.zeros(nhid_cov).astype(np.float32)
covb1=np.zeros(nout_cov).astype(np.float32)
covb2=np.zeros(nx).astype(np.float32)

theano_rng = RandomStreams()

init_params=[muW0, muW1, muW2, mub0, mub1, mub2,
			covW0, covW1, covW2, covb0, covb1, covb2]

def whiten(x):
	mu=np.mean(x,axis=0)
	x=x-mu
	cov=np.cov(x.T)
	cov_inv=np.linalg.inv(cov)
	cov_inv_sqrt=sp.linalg.sqrtm(cov_inv)
	out=np.dot(x,cov_inv_sqrt)
	return out


def compute_f_mu(x, t, params):
	[muW0, muW1, muW2, mub0, mub1, mub2]=params
	h=T.nnet.sigmoid(T.dot(x,muW0)+mub0) #nt by nb by nhidmu
	h2=T.nnet.sigmoid(T.dot(h,muW1)+mub1)
	z=T.dot(h2,muW2)
	z=T.reshape(z,(t.shape[0],t.shape[1],ntgates,nx))+mub2 #nt by nb by ntgates by nx
	#z=z+T.reshape(x,(t.shape[0],t.shape[1],1,nx))
	
	tpoints=T.cast(T.arange(ntgates),'float32')/T.cast(ntgates-1,'float32')
	tpoints=T.reshape(tpoints, (1,1,ntgates))
	#tgating=T.exp(T.dot(t,muWT)+mubT) #nt by nb by ntgates
	tgating=T.exp(-kT*(tpoints-t)**2)
	tgating=tgating/T.reshape(T.sum(tgating, axis=2),(t.shape[0], t.shape[1], 1))
	tgating=T.reshape(tgating,(t.shape[0],t.shape[1],ntgates,1))
	
	mult=z*tgating
	
	out=T.sum(mult,axis=2)
	
	out=out+x
	
	return T.cast(out,'float32')


def compute_f_cov(x, t, params):
	[covW0, covW1, covW2, covb0, covb1, covb2]=params
	h=T.nnet.sigmoid(T.dot(x,covW0)+covb0) #nt by nb by nhidmu
	h2=T.nnet.sigmoid(T.dot(h,covW1)+covb1)
	z=T.dot(h2,covW2)
	z=T.reshape(z,(t.shape[0],t.shape[1],ntgates,nx))+covb2 #nt by nb by ntgates by 1
	z=T.exp(z)
	
	tpoints=T.cast(T.arange(ntgates),'float32')/T.cast(ntgates-1,'float32')
	tpoints=T.reshape(tpoints, (1,1,ntgates))
	#tgating=T.exp(T.dot(t,covWT)+covbT) #nt by nb by ntgates
	tgating=T.exp(-kT*(tpoints-t)**2)
	tgating=tgating/T.reshape(T.sum(tgating, axis=2),(t.shape[0], t.shape[1], 1))
	tgating=T.reshape(tgating,(t.shape[0],t.shape[1],ntgates,1))
	
	mult=z*tgating
	
	out=T.sum(mult,axis=2)
	
	return T.cast(out,'float32')


def forward_step(x, t):
	
	samps=theano_rng.normal(size=x.shape)*T.sqrt(beta+t)
	means=x*T.sqrt(1.0-(beta+t))
	return T.cast(means+samps,'float32'), T.cast(t+0.00/200.0,'float32')


def compute_forward_trajectory(x0):
	
	[x_seq, ts], updates=theano.scan(fn=forward_step,
									outputs_info=[x0, 0.0],
									n_steps=nsteps)
	return x_seq, updates


def loss(x_seq, t, params):
	muparams=params[:6]
	covparams=params[6:]
	f_mu=compute_f_mu(x_seq,t,muparams)
	f_cov=compute_f_cov(x_seq,t,covparams)
	#f_cov=T.extra_ops.repeat(f_cov,self.nx,axis=2)
	diffs=(f_mu[1:]-x_seq[:-1])**2
	gaussian_terms=T.sum(diffs*(1.0/f_cov[1:]),axis=2)
	det_terms=T.sum(T.log(f_cov[1:]),axis=2)
	
	return gaussian_terms+det_terms


def get_loss_grad(params, x_seq):
	
	t0=T.cast(T.arange(nsteps),'float32')/T.cast(nsteps,'float32')
	t=T.reshape(t0,(nsteps,1,1))
	t=T.extra_ops.repeat(t,batchsize,axis=1)
	loss_terms=loss(x_seq,t,params)
	objective=T.mean(T.mean(loss_terms))
	gparams=T.grad(objective, params, consider_constant=[x_seq,t])
	
	return objective, gparams


def reverse_step(x, t, nsamps, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11):
	
	muparams=[p0, p1, p2, p3, p4, p5]
	covparams=[p6, p7, p8, p9, p10, p11]
	f_mu=compute_f_mu(x,t,muparams)
	f_cov=compute_f_cov(x,t,covparams)
	#f_cov=T.extra_ops.repeat(f_cov,self.nx,axis=2)
	samps=theano_rng.normal(size=(1,nsamps, nx))
	samps=samps*T.sqrt(f_cov)+f_mu
	return samps,T.cast(t-1.0/nsteps,'float32')


def get_samps(nsamps, params):
	
	t=1.0
	t=T.reshape(t,(1,1,1))
	t=T.extra_ops.repeat(t,nsamps,axis=1)
	t=T.cast(t,'float32')
	x0=theano_rng.normal(size=(nsamps, nx))
	x0=T.reshape(x0,(1,nsamps,nx))
	[samphist, ts], updates=theano.scan(fn=reverse_step,
									outputs_info=[x0,t],
									non_sequences=[nsamps,params[0],params[1],params[2],params[3],params[4],params[5],
						params[6],params[7],params[8],params[9],params[10],params[11]],
									n_steps=nsteps+1)
	return samphist[-1,0,:,:], ts[:,0], updates


def get_tgating():
		t0=T.cast(T.arange(nsteps),'float32')/T.cast(nsteps,'float32')
		t=T.reshape(t0,(nsteps,1,1))
		t=T.extra_ops.repeat(t,1,axis=1)
		tpoints=T.cast(T.arange(ntgates),'float32')/T.cast(ntgates-1,'float32')
		tpoints=T.reshape(tpoints, (1,1,ntgates))
		#tgating=T.exp(T.dot(t,muWT)+mubT) #nt by nb by ntgates
		tgating=T.exp(-kT*(tpoints-t)**2)
		tgating=tgating/T.reshape(T.sum(tgating, axis=2),(t.shape[0], t.shape[1], 1))
		tgating=T.reshape(tgating,(t.shape[0],t.shape[1],ntgates,1))
		return tgating

#compute_tgating=theano.function([],get_tgating()[:,0,:,0])

#tgates=compute_tgating()
#print tgates.shape
#pp.plot(tgates)
#pp.figure(2)
#pp.plot(np.sum(tgates,axis=1))
#pp.show()

### Making the swiss roll

#data=np.random.rand(nsamps,2)*8.0+4.0
#data=np.asarray([data[:,0]*np.cos(data[:,0]), data[:,0]*np.sin(data[:,0])])+np.random.randn(2,nsamps)*0.25
#data=4.0*data.T

nmix=2
mixmeans=np.random.randn(nmix,nx)*0.0
mixmeans[0,0]=12.0; mixmeans[1,0]=-12.0#; mixmeans[2,1]=12.0; mixmeans[3,1]=-12.0
probs=np.random.rand(nmix)*0.0+1.0
probs=probs/np.sum(probs)
data=[]
for i in range(nsamps):
	midx=np.dot(np.arange(nmix),np.random.multinomial(1,probs))
	nsamp=np.random.randn(nx)*(float(midx)+1.0)*1.0
	data.append(mixmeans[int(midx)]+nsamp)

data=np.asarray(data, dtype='float32')
data=whiten(data)*1.0

#pp.scatter(data[:,0],data[:,1]); pp.show()

# Computing the forward trajectories and subfunction list

xT=T.fmatrix()
xseq, xseq_updates=compute_forward_trajectory(xT)
get_forward_traj=theano.function([xT],xseq,updates=xseq_updates,allow_input_downcast=True)


if save_forward_animation:
	fdata=get_forward_traj(data)
	fig = pp.figure()
	ax = pp.axes(xlim=(-5, 5), ylim=(-5, 5))
	paths = ax.scatter(fdata[0,:,0],fdata[0,:,1],c='r')

	def init():
		paths.set_offsets(fdata[0,:,:])
		return paths,

	# animation function.  This is called sequentially
	def animate(i):
		if i<nsteps:
			paths.set_offsets(fdata[i,:,:])
		else:
			paths.set_offsets(fdata[-1,:,:])
		return paths,

	anim = animation.FuncAnimation(fig, animate, init_func=init,
								   frames=nsteps+50, interval=20, blit=True)

	mywriter = animation.FFMpegWriter()
	anim.save('forward_process.mp4', fps=20)


subfuncs=[]
endcov=np.zeros((nx,nx))
for i in range(n_subfuncs):
	idxs=np.random.randint(nsamps-1,size=batchsize)
	subfuncs.append(np.asarray(get_forward_traj(data[idxs]),dtype='float32'))
	endcov+=np.cov(subfuncs[i][-1,:,:].T)

print endcov/float(n_subfuncs)
pp.scatter(subfuncs[0][-1,:,0],subfuncs[0][-1,:,1]); pp.show()

# Compiling the loss and gradient function

xtrajT=T.ftensor3()
[muW0T, muW1T, muW2T, mub0T, mub1T, mub2T,
	covW0T, covW1T, covW2T, covb0T, covb1T, covb2T]=[T.fmatrix(), T.fmatrix(), T.fmatrix(), 
			T.fvector(), T.fvector(), T.fvector(),
			T.fmatrix(), T.fmatrix(), T.fmatrix(), 
			T.fvector(), T.fvector(), T.fvector()]

paramsT=[muW0T, muW1T, muW2T, mub0T, mub1T, mub2T,
		covW0T, covW1T, covW2T, covb0T, covb1T, covb2T]

lossT, gradT=get_loss_grad(paramsT, xtrajT)

f_df_T=theano.function([muW0T, muW1T, muW2T, mub0T, mub1T, mub2T, 
					covW0T, covW1T, covW2T, covb0T, covb1T, covb2T, xtrajT],
					[lossT,gradT[0],gradT[1],gradT[2],gradT[3],gradT[4],gradT[5],
					gradT[6],gradT[7],gradT[8],gradT[9],gradT[10],gradT[11],],
					allow_input_downcast=True,
					on_unused_input='warn')

def f_df(params, subfunc):
	[loss, grad0,grad1,grad2,grad3,grad4,grad5,
	grad6,grad7,grad8,grad9,grad10,grad11] = f_df_T(params[0],params[1],params[2],params[3],params[4],params[5],
						params[6],params[7],params[8],params[9],params[10],params[11],
						subfunc)
	return loss, [grad0,grad1,grad2,grad3,grad4,grad5,grad6,grad7,grad8,grad9,grad10,grad11]


# Compiling the sampling function

samplesT, tT, sample_updates=get_samps(nsamps, paramsT)
sample_T=theano.function([muW0T, muW1T, muW2T, mub0T, mub1T, mub2T, 
					covW0T, covW1T, covW2T, covb0T, covb1T, covb2T],
					samplesT,
					allow_input_downcast=True)

def sample(params):
	out = sample_T(params[0],params[1],params[2],params[3],params[4],params[5],
						params[6],params[7],params[8],params[9],params[10],params[11])
	return out

# Creating the optimizer

optimizer = SFO(f_df, init_params, subfuncs)

# Running the optimization

init_loss = f_df(init_params,subfuncs[0])[0]
print init_loss

keyin=''
while keyin!='y':
	opt_params = optimizer.optimize(num_passes=24*4)
	end_loss = f_df(opt_params,subfuncs[0])[0]
	print 'Current loss: ', end_loss
	W=opt_params[0]
	pp.scatter(W[0,:],W[1,:]); pp.show()
	keyin=raw_input('End optimization? (y)')

samples=sample(opt_params)
pp.scatter(samples[:,0],samples[:,1]); pp.show()
