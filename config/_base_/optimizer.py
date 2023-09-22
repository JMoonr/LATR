# opt setting
optimizer = 'adam'
learning_rate = 2e-4

weight_decay = 0.001
lr_decay = False # TODO 'store_true'
niter = 900 # num of iter at starting learning rate
niter_decay = 400 # '# of iter to linearly decay learning rate to zero'
lr_policy = 'cosine'
gamma = 0.1 # multiplicative factor of learning rate decay
lr_decay_iters = 10 # multiply by a gamma every lr_decay_iters iterations
T_max = 8 # maximum number of iterations
T_0 = 8
T_mult = 2
eta_min = 1e-5 # minimum learning rate
clip_grad_norm = 35.0 # grad clipping
loss_threshold = 1e5

