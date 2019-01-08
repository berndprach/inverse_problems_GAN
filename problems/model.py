from __future__ import division
import os
import time
import math
import glob
import tensorflow as tf
import numpy as np
import importlib
import pickle
from datetime import datetime

import ops
import utils

class DCGAN(object):
  def __init__(self, sess, args,
                batch_size=64, nrof_samples = 64,
                gf_dim=64, df_dim=64,
                gfc_dim=1024, dfc_dim=1024, c_dim=3):
    
    """
    Args:
      sess: TensorFlow session
      args: Values for most things
      batch_size: The size of batch. Should be specified before training.
      nrof_samples: The number of samples. Should be specified before training.
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    
    self.sess = sess
    self.args = args

    self.args.batch_size = batch_size
    self.args.nrof_samples = nrof_samples
    
    self.args.gf_dim = gf_dim
    self.args.df_dim = df_dim

    self.args.gfc_dim = gfc_dim
    self.args.dfc_dim = dfc_dim

    # Import Problem Modul:
    self.Problem = importlib.import_module(args.problem_name)
    
    self.prepare_dataset()
        
    self.args.grayscale = (self.args.c_dim == 1)
    
    self.args.g_input_dim = [self.args.output_height, self.args.output_width,self.args.c_dim]

    self.build_model()

  def build_model(self):
    if self.args.crop:
      image_dims = [self.args.output_height, self.args.output_width, self.args.c_dim]
    else:
      image_dims = [self.args.input_height, self.args.input_width, self.args.c_dim]

    self.d_inputs = tf.placeholder(
      tf.float32, [self.args.batch_size] + image_dims, name='real_images')
    
    self.z = tf.placeholder(
      tf.float32, [None, self.args.z_dim], name='z')
    self.g_inputs = tf.placeholder(
      tf.float32, [self.args.batch_size] + self.args.g_input_dim, name='g_inputs')
    #self.z_sum = ops.histogram_summary("z", self.z)
    
    self.g_tform_info = self.Problem.g_tform_info_placeholder

    self.G                  = self.generator(self.z, self.g_inputs)
    self.D, self.D_logits   = self.discriminator(self.d_inputs, reuse=False)
    self.sampler            = self.generator(self.z, self.g_inputs, sampler=True)
    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
    
    #self.d_sum = ops.histogram_summary("d", self.D)
    #self.d__sum = ops.histogram_summary("d_", self.D_)
    #self.G_sum = ops.image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      
      
    self.G_tformed = self.Problem.transform_tf(self.G, self.g_tform_info)
    self.sampler_tformed = self.Problem.transform_tf(self.sampler, self.g_tform_info)
    
    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D))) # = -log(sigmoid( D_logits ))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_))) # = -log(1 - sigmoid( D_logits_ ))
    # Normalising (should hold): errD <= log(4) ~ 1.39 (= error for random guessing)
    self.d_loss = (self.d_loss_real + self.d_loss_fake) / np.log(4)
    
    self.g_disc_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_))) # = -log(sigmoid( D_logits_ ))
    self.g_disc_loss = (self.g_disc_loss - np.log(2)) / np.log(4)
    self.g_prob_loss = tf.reduce_mean(self.Problem.problem_loss(self.g_inputs, self.G_tformed))
    assert self.g_prob_loss.dtype == tf.float32
    self.g_loss = self.g_disc_loss + self.args.lambda_loss * self.g_prob_loss
    
    #self.d_loss_real_sum = ops.scalar_summary("d_loss_real", self.d_loss_real)
    #self.d_loss_fake_sum = ops.scalar_summary("d_loss_fake", self.d_loss_fake)
                         
    #self.g_loss_sum = ops.scalar_summary("g_loss", self.g_loss)
    #self.d_loss_sum = ops.scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self):
    d_optim = tf.train.AdamOptimizer(self.args.learning_rate, beta1=self.args.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(self.args.learning_rate, beta1=self.args.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
              
    self.sess.run(tf.global_variables_initializer(), feed_dict=None)
    
    #self.g_sum = ops.merge_summary([self.z_sum, self.d__sum,
    #  self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    #self.d_sum = ops.merge_summary(
    #    [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    #self.writer = ops.SummaryWriter("./logs", self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.args.nrof_samples , self.args.z_dim))
        
    sample_images = utils.get_img(self, 0, self.args.nrof_samples, self.args.dataset_name, test=False)
    test_images = utils.get_img(self, 0, self.args.nrof_samples, self.args.dataset_name, test=True)
    
    sample_tform_info = self.Problem.create_tform_info(self.args)
    test_tform_info = self.Problem.create_tform_info(self.args)
    
    sample_g_inputs = self.Problem.transform(sample_images,sample_tform_info)
    test_g_inputs = self.Problem.transform(test_images,test_tform_info)

    utils.save_images(sample_images,'samples\original_images_s.png')
    utils.save_images(test_images,'samples\original_images_t.png')
    
    utils.save_images(self.Problem.safe_format(sample_g_inputs),'samples\original_inputs_s.png')
    utils.save_images(self.Problem.safe_format(test_g_inputs),'samples\original_inputs_t.png')
    
    sample_dict={self.z: sample_z, #For generator and discriminator
                self.g_inputs: sample_g_inputs,
                self.d_inputs: sample_images,
                self.g_tform_info: sample_tform_info}
    test_dict={self.z: sample_z,
                self.g_inputs: test_g_inputs,
                self.d_inputs: test_images,
                self.g_tform_info: test_tform_info}          
    
    #Set up for visualizing difference from z value
    z_range = utils.get_z_range(self.args.z_dim, self.args.batch_size)    

    nrof_batches = self.args.train_size // self.args.batch_size
      
    counter = 0
    start_epoch_nr = 0
    start_batch_nr = 0
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.args.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      start_epoch_nr = checkpoint_counter // nrof_batches
      start_batch_nr = checkpoint_counter % nrof_batches  
      print(" [*] Load SUCCESS (counter: " + str(counter) + ")")
      with open(self.args.progress_file_name,"a") as prog_file:
        prog_file.write("\n" + datetime.now().strftime("%H:%M:%S ") + \
                        "Loaded checkpoint " + str(counter) + "\n")
    else:
      print(" [!] Load failed...")
      with open(self.args.progress_file_name,"w") as prog_file:
        prog_file.write("\n" + datetime.now().strftime("%H:%M:%S ") + \
                        "No checkpoint found, starting training from scratch.\n")
    for epoch0 in range(self.args.nrof_epochs):
      epoch = start_epoch_nr + epoch0
      
      with open(self.args.progress_file_name,'a') as prog_file:
        prog_file.write("\n" + datetime.now().strftime("%H:%M:%S ") + \
                        "Started training epoch " + str(epoch) + "\n")
    
      for idx0 in range(start_batch_nr,nrof_batches):      
        idx = idx0+1
        
        batch_images = utils.get_img(self, idx0*self.args.batch_size,
                          self.args.batch_size, self.args.dataset_name, test=False)
        
        batch_tform_info = self.Problem.create_tform_info(self.args)
        batch_g_inputs = self.Problem.transform(batch_images, batch_tform_info)
        batch_z = np.random.uniform(-1, 1, [self.args.batch_size, self.args.z_dim]).astype(np.float32)

        D_dict = {self.d_inputs: batch_images,
                  self.z: batch_z,
                  self.g_inputs: batch_g_inputs}
        G_dict = {self.z: batch_z, 
                  self.g_inputs: batch_g_inputs,
                  self.g_tform_info: batch_tform_info}
        

        # Update D network
        _, err_D, err_G, err_G_disc, err_G_prob = \
              self.sess.run([d_optim, self.d_loss, self.g_loss,self.g_disc_loss,self.g_prob_loss],
                            feed_dict={**D_dict,**G_dict})
        #self.writer.add_summary(summary_str, counter)

        # Update G network
        self.sess.run(g_optim, feed_dict=G_dict)
        #self.writer.add_summary(summary_str, counter)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        self.sess.run(g_optim, feed_dict=G_dict)
        #self.writer.add_summary(summary_str, counter)

        counter += 1
        time_str = time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time))
        print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {}, d_loss: {:6.4f}, g_loss: {:6.4f}" \
          .format(epoch, idx, nrof_batches, time_str, err_D, err_G))
        #Should hold: errD <= log(4) ~ 1.39 (= error for random guessing)
        
        if np.mod(counter, self.args.save_freq) == 0:
          print("g_loss: {:.8f} (D) + {:g} * {:.8f} (problem) = {:.8f}".\
                format(err_G_disc, self.args.lambda_loss, err_G_prob, err_G))
        
          samples, samples_tformed, d_loss, g_loss = self.sess.run(
            [self.sampler, self.sampler_tformed, self.d_loss, self.g_loss], feed_dict=sample_dict)
          utils.save_images(samples, '{}/train_{:02d}_{:04d}_samples_s.png'.format(self.args.sample_dir, epoch, idx))
          print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
          
          test_samples, test_samples_tformed, test_d_loss, test_g_loss = self.sess.run(
            [self.sampler, self.sampler_tformed, self.d_loss, self.g_loss], feed_dict=test_dict)
          utils.save_images(test_samples,'{}/train_{:02d}_{:04d}_samples_t.png'.format(self.args.sample_dir, epoch, idx))
          print("[Test] d_loss: %.8f, g_loss: %.8f" % (test_d_loss, test_g_loss)) 

          with open(self.args.progress_file_name,'a') as prog_file:
            out_str = "Epoch: [{:2d}] [{:4d}/{:4d}] ".format(epoch, idx, nrof_batches) + \
                      "\td_loss: {:6.4f} ".format(err_D) + \
                      "\tg_loss: {:6.4f} = {:.8f} (D) + {:g} * {:.8f} (problem)" \
                      .format(err_G, err_G_disc, self.args.lambda_loss, err_G_prob)
            prog_file.write(out_str + "\n")
          
        if np.mod(counter, 5*self.args.save_freq) == 0:
          self.save(self.args.checkpoint_dir, counter)
          
          utils.save_multiple(2,[samples, sample_images], 'train_{:02d}_{:04d}_comp'.format(epoch, idx))
          save_pics = [sample_images, self.Problem.safe_format(sample_g_inputs), \
                        samples, self.Problem.safe_format(samples_tformed)]
          utils.save_multiple(4, save_pics, 'train_{:02d}_{:04d}_ovw_s'.format(epoch, idx))
          

          utils.save_multiple(2,[test_samples, test_images], 'train_{:02d}_{:04d}_comp_test'.format(epoch, idx))
          save_pics = [test_images, self.Problem.safe_format(test_g_inputs), \
                        test_samples, self.Problem.safe_format(test_samples_tformed)]
          utils.save_multiple(4, save_pics, 'train_{:02d}_{:04d}_ovw_t'.format(epoch, idx))

          print("Checkpoint!")
          
          #Visualize change with z:
          print("visualizing for different z values ...")
          for i in range(2):          
            input_idx = np.random.randint(self.args.batch_size)
            
            vis_z_g_inputs = np.repeat([test_g_inputs[input_idx]],self.args.batch_size,axis=0)
            vis_z_images = np.repeat([test_images[input_idx]],self.args.batch_size,axis=0)
            vis_z_tform_info = np.repeat([test_tform_info[input_idx]],self.args.batch_size,axis=0)
            
            vis_z_dict={self.z: z_range,
                        self.g_inputs: vis_z_g_inputs,
                        self.d_inputs: vis_z_images,
                        self.g_tform_info: vis_z_tform_info}
        
            vis_z_samples = self.sess.run(self.sampler, feed_dict=vis_z_dict)
            vis_z_merged = self.Problem.merge(vis_z_samples, vis_z_g_inputs, vis_z_tform_info)
            utils.save_images(vis_z_merged, '{}/train_{:02d}_{:04d}_vis_z_{:01d}.png'.format(self.args.sample_dir, epoch, idx, input_idx))
      
            print("Mean Standard deviation: " + str(np.mean(np.std(samples, axis=0))))
            
            with open(self.args.progress_file_name,'a') as prog_file:
              prog_file.write("\tMean Standard deviation: " + \
                              str(np.mean(np.std(samples, axis=0))) + "\n")
              
      
      #Visualize at the end of every epoch
      if epoch0<8:
        for i in range(8):
          for j in range(8):
            pic_idx = 8*i + j
            utils.save_images(test_samples[pic_idx:pic_idx+1:],
                    'samples_progress/part{:01d}/pic{:02d}_epoch{:02d}.jpg'.format(i+1, pic_idx, epoch))
      
      start_batch_nr = 0
      
    #save a final checkpoint
    self.save(self.args.checkpoint_dir, counter)
    
    with open(self.args.progress_file_name,'a') as prog_file:
      prog_file.write("\n" + datetime.now().strftime("%H:%M:%S ") + "Finished training." + "\n")
      
  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = ops.lrelu(ops.conv2d(image, self.args.df_dim, name='d_h0_conv'))
      h1 = ops.lrelu(ops.bn_layer(ops.conv2d(h0, self.args.df_dim*2, name='d_h1_conv'), name="d_bn1"))
      h2 = ops.lrelu(ops.bn_layer(ops.conv2d(h1, self.args.df_dim*4, name='d_h2_conv'), name="d_bn2"))
      h3 = ops.lrelu(ops.bn_layer(ops.conv2d(h2, self.args.df_dim*8, name='d_h3_conv'), name="d_bn3"))
      h4 = ops.linear(tf.reshape(h3, [self.args.batch_size, -1]), 1, 'd_h4_lin')

      return tf.nn.sigmoid(h4), h4
        
  def generator(self, z, g_inputs, y=None, sampler=False):
    with tf.variable_scope("generator") as scope:
      if sampler:
        scope.reuse_variables()
      do_train = not sampler
      
      bs = self.args.batch_size
      
      conv_out_size_same = lambda h, w, stride: [ int(math.ceil(s/stride)) for s in [h,w] ]
      
      s_h, s_w = self.args.output_height, self.args.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, s_w8, 2)
      
      # *** First layers: g_inputs => g_flat *** #   
      gi = ops.lrelu(ops.conv2d(g_inputs, self.args.df_dim, name='g_gi0_conv'))
      for idx in range(1,4):
        conv = ops.conv2d(gi, self.args.df_dim*(2**idx), name="g_gi" + str(idx) + "_conv")
        gi = ops.lrelu(ops.bn_layer(conv,train=do_train, name="gi" + str(idx) + "_bn"))
      gi_flat = ops.linear(tf.reshape(gi, [bs, -1]), self.args.g_feature_dim, 'g_gi4_lin')
      
      # *** Map gi_flat to [-1,1] to be more similar to z: *** #
      gi_flat = tf.nn.tanh(gi_flat)
        
      # *** Layers from flat (z and gi_flat) to full size: *** #
      z0 = ops.concat( [gi_flat, z], -1 )

      gd0 = ops.linear( z0, self.args.gf_dim*8*s_h16*s_w16, 'g_h0_lin')
      gd0 = tf.reshape( gd0, [bs, s_h16, s_w16, self.args.gf_dim * 8])
      gd0 = tf.nn.relu(ops.bn_layer(gd0, train=do_train, name="g_bn0"))
      
      gd = gd0
      s = [None, s_h8,s_h4,s_h2,s_h]
      m = [None, 4,2,2,2]
      for idx in range(1,5):
        deconv = ops.deconv2d(gd,[bs,s[idx],s[idx],self.args.gf_dim*m[idx]],name="g_h"+str(idx))
        gd = tf.nn.relu(ops.bn_layer(deconv, train=do_train, name="g_bn"+str(idx)))
      gd4 = ops.concat( [ gd, g_inputs], -1)
      
      # *** 2 Layers to merge gd and g_inputs: *** #   
      gd5 = ops.deconv2d(gd4, [bs, s_h, s_w, self.args.gf_dim], k_h = 1, k_w = 1, d_h=1, d_w=1, name='g_h5')
      gd5 = tf.nn.relu(gd5)
      gd6 = ops.deconv2d(gd5, [bs, s_h, s_w, self.args.c_dim], k_h = 1, k_w = 1, d_h=1, d_w=1, name='g_h6')

      return tf.nn.sigmoid(gd6)
  
  def prepare_dataset(self):
    #if "jpg" in self.input_fname_pattern or "png" in self.input_fname_pattern:
    if self.args.dataset_name == "celebA":
      data_paths = glob.glob(os.path.join(self.args.data_dir, self.args.dataset_name, self.args.input_fname_pattern))
      imreadImg = utils.imread(data_paths[0])
      if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
        self.args.c_dim = imreadImg.shape[-1]
      else:
        self.args.c_dim = 1
      train_size = len(data_paths)-1000
      if self.args.train_size is not None:
        self.args.train_size = min(self.args.train_size,train_size)
      else:
        self.args.train_size = train_size
      self.data_paths = data_paths[1000:1000+self.args.train_size]
      self.data_paths_val = data_paths[:1000]    
    #elif "ubyte" in self.input_fname_pattern
    elif self.args.dataset_name == "mnist" or self.args.dataset_name == "cifar10":
      if self.args.dataset_name == "mnist":
        self.data_X, self.data_X_val = self.load_mnist()
      elif self.args.dataset_name == "cifar10":
        #data = self.load_cifar10_batch(1)
        #self.data_X_val, self.data_X = data[:100], data[100:]        
        self.data_X = self.load_cifar10_batch(1)
        for i in range(2,6):
          self.data_X = np.concatenate((self.data_X, self.load_cifar10_batch(i)), axis=0)
        self.data_X_val = self.load_cifar10_batch("test")
      if self.args.train_size is not None:
        self.data_X = self.data_X[:self.args.train_size]
      self.args.train_size = len(self.data_X)
      self.args.c_dim = self.data_X[0].shape[-1]         
  
  def load_mnist(self):
    data_dir = os.path.join(self.args.data_dir, self.args.dataset_name)

    with open(os.path.join(data_dir,'train-images-idx3-ubyte')) as train_file:
      loaded = np.fromfile(file=train_file,dtype=np.uint8)
      trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    with open(os.path.join(data_dir,'t10k-images-idx3-ubyte')) as test_file:
      loaded = np.fromfile(file=test_file,dtype=np.uint8)
      teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    #X = np.concatenate((trX, teX), axis=0)

    seed = 547
    np.random.seed(seed)
    #np.random.shuffle(X)
    np.random.shuffle(trX)
    np.random.shuffle(teX)

    return trX/255., teX/255. 
   
  def load_cifar10_batch(self, batch_id):
    data_dir = os.path.join(self.args.data_dir, self.args.dataset_name, "cifar-10-batches-py")
    batch_name = "test_batch" if batch_id == "test" else "data_batch_" + str(batch_id)
    with open(os.path.join(data_dir,batch_name), mode='rb') as file:
      batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    features = np.array(features).astype(np.float)
    labels = batch['labels']
    
    labels = np.array(labels)
    
    good_labels = [0,1,5,7,8] #plane, car, dog, horse, ship
    features = features[np.isin(labels, good_labels)]
    
    return features/255.
   
    
  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.args.dataset_name, self.args.batch_size,
        self.args.output_height, self.args.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
