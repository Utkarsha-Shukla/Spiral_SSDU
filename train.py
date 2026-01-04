import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
from datetime import datetime
import os
import h5py as h5
import utils
import tf_utils
import parser_ops
import masks.ssdu_masks as ssdu_masks
import UnrollNet
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)  #US--OOM

parser = parser_ops.get_parser()
args, _ = parser.parse_known_args()


save_dir ='saved_models'
directory = os.path.join(save_dir, 'SSDU_' + args.data_opt + '_' +str(args.epochs)+'Epochs_Rate'+ str(args.acc_rate) +\
                         '_' + str(args.nb_unroll_blocks) + 'Unrolls_' + args.mask_type+'Selection' )

log_dir = os.path.join("/home/hpc/mfdp/mfdp120h/SSDU-main-GIT/code/saved_models/tensorboard",datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)
# summary_writer = tf.compat.v1.summary.create_file_writer(log_dir)

if not os.path.exists(directory):
    os.makedirs(directory)

print('\n create a test model for the testing')
test_graph_generator = tf_utils.test_graph(directory)

#...........................................................................d....
start_time = time.time()
print('.................SSDU Training.....................')
tf.compat.v1.reset_default_graph()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

# .......................Load the Data..........................................
print('\n Loading ', args.data_opt, ' data, acc rate : ', args.acc_rate, ', mask type :', args.mask_type)
kspace_dir, coil_dir, mask_dir = utils.get_train_directory(args)

# %% kspace and sensitivity maps are assumed to be in .h5 format and mask is assumed to be in .mat
# Users can change these formats based on their dataset
kspace_train = h5.File(kspace_dir, "r")['kspace'][:]  #[slices, coil, h, w]
sens_maps = h5.File(coil_dir, "r")['sens_maps'][:]  #[slices, coil, h, w]
original_mask = sio.loadmat(mask_dir)['mask'] #[w,h]
original_mask = np.transpose(original_mask, (1, 0)) #for consistency [h,w]

#Oversampling in the PE direction
print("############################# Cropping ###########################")
kx = kspace_train.shape[2]
new_kx = kx //2  # e.g., 768 // 2 = 384
start = (kx - new_kx) // 2
end = start + new_kx
kspace_train = kspace_train[:, :, start:end, :]
sens_maps=sens_maps[:, :, start:end, :]
original_mask=original_mask[start:end,:]
print("kspace_train", kspace_train.shape, "sens_maps", sens_maps.shape, "original_mask", original_mask.shape)

kspace_train=np.transpose(kspace_train,(0,2,3,1)) #[nslc, ncoil, nrow, ncol]-->[nslc, nrow, ncol, ncoil]
sens_maps = np.transpose(sens_maps,(0,2,3,1)) #[nslc, ncoil, nrow, ncol]-->[nslc, nrow, ncol, ncoil]

print('\n Normalize the kspace to 0-1 region')
for ii in range(np.shape(kspace_train)[0]):
    kspace_train[ii, :, :, :] = kspace_train[ii, :, :, :] / np.max(np.abs(kspace_train[ii, :, :, :][:]))

print('\n size of kspace: ', kspace_train.shape, ', maps: ', sens_maps.shape, ', mask: ', original_mask.shape)
nSlices, *_ = kspace_train.shape
trn_mask, loss_mask = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64), \
                      np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

nw_input = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
ref_kspace = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)

print('\n create training and loss masks and generate network inputs... ')
ssdu_masker = ssdu_masks.ssdu_masks()
for ii in range(nSlices):
    if np.mod(ii, 50) == 0:
        print('\n Iteration: ', ii)

    if args.mask_type == 'Gaussian':
        trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.Gaussian_selection(kspace_train[ii], original_mask, num_iter=ii)

    elif args.mask_type == 'Uniform':
        trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.uniform_selection(kspace_train[ii], original_mask, num_iter=ii)

    else:
        raise ValueError('Invalid mask selection')

    sub_kspace = kspace_train[ii] * np.tile(trn_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    ref_kspace[ii, ...] = kspace_train[ii] * np.tile(loss_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    nw_input[ii, ...] = utils.sense1(sub_kspace, sens_maps[ii, ...])

# %%  zeropadded outer edges of k-space with no signal- check github readme file for explanation for further explanations
# for coronal PD dataset, first 17 and last 16 columns of k-space has no signal
# in the training mask we set corresponding columns as 1 to ensure data consistency
if args.data_opt == 'Coronal_PD':
    trn_mask[:, :, 0:17] = np.ones((nSlices, args.nrow_GLOB, 17))
    trn_mask[:, :, 352:args.ncol_GLOB] = np.ones((nSlices, args.nrow_GLOB, 16))

# %% Prepare the data for the training
sens_maps = np.transpose(sens_maps, (0, 3, 1, 2))
ref_kspace = utils.complex2real(np.transpose(ref_kspace, (0, 3, 1, 2)))
nw_input = utils.complex2real(nw_input)

print('\n size of ref kspace: ', ref_kspace.shape, ', nw_input: ', nw_input.shape, ', maps: ', sens_maps.shape, ', mask: ', trn_mask.shape)

# %% set the batch size
total_batch = int(np.floor(np.float32(nw_input.shape[0]) / (args.batchSize)))
kspaceP = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None, None, 2), name='refkspace')
sens_mapsP = tf.compat.v1.placeholder(tf.complex64, shape=(None, None, None, None), name='sens_maps')
trn_maskP = tf.compat.v1.placeholder(tf.complex64, shape=(None, None, None), name='trn_mask')
loss_maskP = tf.compat.v1.placeholder(tf.complex64, shape=(None, None, None), name='loss_mask')
nw_inputP = tf.compat.v1.placeholder(tf.float32, shape=(None, args.nrow_GLOB, args.ncol_GLOB, 2), name='nw_input')


# %% creating the dataset
dataset = tf.data.Dataset.from_tensor_slices((kspaceP, nw_inputP, sens_mapsP, trn_maskP, loss_maskP))
dataset = dataset.shuffle(buffer_size=10 * args.batchSize)
dataset = dataset.batch(args.batchSize)
dataset = dataset.prefetch(args.batchSize)

iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
ref_kspace_tensor, nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor = iterator.get_next()

# %% make training model
model_un = UnrollNet.UnrolledNet(
    nw_input_tensor,
    sens_maps_tensor,
    trn_mask_tensor,
    loss_mask_tensor
)
nw_output_img, nw_output_kspace, *_ = model_un.model

print("nw_output_img", nw_output_img.shape)
print("nw_output_kspace", nw_output_kspace.shape)

scalar = tf.constant(0.5, dtype=tf.float32)
loss = tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace) / tf.norm(ref_kspace_tensor)) + \
       tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace, ord=1) / tf.norm(ref_kspace_tensor, ord=1))
print("loss", loss.shape)

#all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
all_trainable_vars = tf.reduce_sum([tf.size(v) for v in tf.compat.v1.trainable_variables()])
update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

saver = tf.compat.v1.train.Saver(max_to_keep=100)
sess_trn_filename = os.path.join(directory, 'model')
totalLoss = []
avg_cost = 0

#Placeholders for logging
input_img_ph  = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1])
output_img_ph = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1])

input_summary  = tf.compat.v1.summary.image("Input",  input_img_ph,  max_outputs=4)
output_summary = tf.compat.v1.summary.image("Output", output_img_ph, max_outputs=4)

merged_summary = tf.compat.v1.summary.merge([input_summary, output_summary])

vis_batch_size = min(4, args.batchSize)

vis_feed = {
    kspaceP: ref_kspace[:vis_batch_size],
    nw_inputP: nw_input[:vis_batch_size],
    trn_maskP: trn_mask[:vis_batch_size],
    loss_maskP: loss_mask[:vis_batch_size],
    sens_mapsP: sens_maps[:vis_batch_size]
}


with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print('SSDU Parameters: Epochs: ', args.epochs, ', Batch Size:', args.batchSize,
          ', Number of trainable parameters: ', sess.run(all_trainable_vars))
    feedDict = {kspaceP: ref_kspace, nw_inputP: nw_input, trn_maskP: trn_mask, loss_maskP: loss_mask, sens_mapsP: sens_maps}

    summary_writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
    
    print('Training...')
    for ep in range(1, args.epochs + 1):
        sess.run(iterator.initializer, feed_dict=feedDict)
        avg_cost = 0
        tic = time.time()
        try:

            for jj in range(total_batch):
                tmp, _, _ = sess.run([loss, update_ops, optimizer])
                avg_cost += tmp / total_batch
            toc = time.time() - tic
            totalLoss.append(avg_cost)
            print("Epoch:", ep, "elapsed_time =""{:f}".format(toc), "cost =", "{:.3f}".format(avg_cost))

        except tf.compat.v1.errors.OutOfRangeError:
            pass

        sess.run(iterator.initializer, feed_dict=feedDict)
        nw_out_vis = sess.run(nw_output_img, feed_dict=vis_feed)
        nw_in_vis  = vis_feed[nw_inputP]
        print("nw_in_vis", nw_in_vis.shape, nw_in_vis.dtype, "nw_out_vis", nw_out_vis.shape,  nw_out_vis.dtype)
        # print("nw_out_vis", nw_out_vis.shape, "nw_out_vis", nw_out_vis.shape)
        # print("After squeeze..................nw_in_vis", nw_in_vis.shape, "nw_out_vis", nw_out_vis.shape)
        
        # Use magnitude for complex data
        nw_in_vis  = np.sqrt(nw_in_vis[..., 0]**2 + nw_in_vis[..., 1]**2)
        nw_out_vis = np.sqrt(nw_out_vis[..., 0]**2 + nw_out_vis[..., 1]**2)


        nw_in_vis  = np.clip(nw_in_vis,  0.0, 1.0)
        nw_out_vis = np.clip(nw_out_vis, 0.0, 1.0)

        # Add channel dimension for TensorBoard
        nw_in_vis  = nw_in_vis[..., np.newaxis]
        nw_out_vis = nw_out_vis[..., np.newaxis]

        summary = sess.run(merged_summary,feed_dict={input_img_ph:  nw_in_vis, output_img_ph: nw_out_vis})  
        summary_writer.add_summary(summary, global_step=ep)
        summary_writer.flush()

        if (np.mod(ep, 10) == 0):
            saver.save(sess, sess_trn_filename, global_step=ep)
            sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'loss': totalLoss})

end_time = time.time()
sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'loss': totalLoss})
print('Training completed in  ', ((end_time - start_time) / 60), ' minutes')
