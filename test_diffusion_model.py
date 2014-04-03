import numpy as np
import theano
import theano.tensor as T
from matplotlib import pyplot as pp
from matplotlib import animation
from diffusion_model import diffusion_model
from matplotlib.path import Path


nx=2
nsamps=64000

nhid_mu=32
nhid_cov=32
nout_mu=16
nout_cov=16

#beta=1e-2
nsteps=200
beta = 1. - np.exp(np.log(0.001)/float(nsteps))
#beta=0.04
print beta

batchsize=10

lrate=1e-3

#making some data
nmix=2
mixmeans=np.random.randn(nmix,nx)*0.0
mixmeans[0,0]=12.0; mixmeans[1,0]=-12.0#; mixmeans[2,1]=12.0; mixmeans[3,1]=-12.0
probs=np.random.rand(nmix)*0.0+1.0
probs=probs/np.sum(probs)
data=[]
for i in range(nsamps):
	midx=np.dot(np.arange(nmix),np.random.multinomial(1,probs))
	nsamp=np.random.randn(nx)*1.0
	data.append(mixmeans[int(midx)]+nsamp)
	
#data=np.random.rand(nsamps,2)*10.0+8.0
#data=np.asarray([data[:,0]*np.cos(data[:,0]), data[:,0]*np.sin(data[:,0])])+np.random.randn(2,nsamps)*0.5
#data=data.T

data=np.asarray(data, dtype='float32')


#data=4.0*data/np.sqrt(np.mean(np.sum(data**2,axis=1)))
print data.shape
pp.scatter(data[:,0],data[:,1]); pp.show()

model=diffusion_model(nx, batchsize, nsteps, beta, nhid_mu, nhid_cov, nout_mu, nout_cov, ntgates=20)

xT=T.fmatrix()
xseq, xseq_updates=model.compute_forward_trajectory(xT)
get_forward_traj=theano.function([xT],xseq,updates=xseq_updates,allow_input_downcast=True)
data_forward_traj=np.asarray(get_forward_traj(data),dtype='float32')
dft_shared=theano.shared(data_forward_traj)
print data_forward_traj.shape
pp.scatter(data_forward_traj[-1,:,0],data_forward_traj[-1,:,1],c='r'); pp.show()

lrT=T.fscalar()
xtrajT=T.ftensor3()

tloss, lterms, learn_updates=model.train_step_nosample(xtrajT, lrT)

idxT=T.lscalar()
train_model=theano.function([idxT,lrT],[tloss,lterms],updates=learn_updates,
						givens={xtrajT: dft_shared[:,idxT:idxT+batchsize,:]},
						allow_input_downcast=True,
						on_unused_input='warn')

gloss=model.get_loss(xtrajT)
compute_loss=theano.function([idxT],gloss,
						givens={xtrajT: dft_shared[:,idxT:idxT+batchsize,:]},
						allow_input_downcast=True,
						on_unused_input='warn')
						
nsampsT=T.lscalar()
tsamps, ts, samp_updates=model.get_samps(nsampsT)
sample_model=theano.function([nsampsT],[tsamps,ts],updates=samp_updates,
						allow_input_downcast=True)

tgt=model.get_tgating()
get_tgates=theano.function([],tgt)

tgates=get_tgates()
#pp.plot(tgates[:,0,:,0]); pp.show()

loss_hist=[]
for i in range(8000):
	idx=np.random.randint(nsamps-batchsize-1)
	batchloss,lossterms=train_model(idx,lrate)
	loss_hist.append(batchloss)
	if i%100==0:
		print '===================================='
		print 'Iter: ',i
		print lrate
		print batchloss-compute_loss(idx)
		print batchloss
		
		#pp.plot(lossterms); pp.show()
	lrate=lrate*0.9997
	#pp.plot(np.asarray(lseq)); pp.show()

loss_hist=np.asarray(loss_hist)
pp.plot(loss_hist); pp.show()

samples,t=sample_model(1000)

fig = pp.figure()
ax = pp.axes(xlim=(-20, 20), ylim=(-20, 20))
paths = ax.scatter(samples[0,:,0],samples[0,:,1],c='r')

def init():
	paths.set_offsets(samples[0,:,:])
	return paths,

# animation function.  This is called sequentially
def animate(i):
	if i<nsteps:
		paths.set_offsets(samples[i,:,:])
	else:
		paths.set_offsets(samples[-1,:,:])
	return paths,

anim = animation.FuncAnimation(fig, animate, init_func=init,
							   frames=nsteps+50, interval=20, blit=True)

anim.save('basic_animation.mp4', fps=20)#, extra_args=['-vcodec', 'libx264'])

pp.show()


#pp.figure(1)
#pp.scatter(data[:,0],data[:,1])
#pp.scatter(samples[:,0],samples[:,1],c='r')
#tgates=get_tgates()
#pp.figure(2); pp.plot(tgates[:,0,:,0])
#pp.show()

