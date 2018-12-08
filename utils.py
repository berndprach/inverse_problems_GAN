"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import scipy.misc
import os
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
from glob import glob

import tensorflow as tf
import tensorflow.contrib.slim as slim

import string
import random

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def save_images(images, image_path, size=None, out_dim = None):
  nrof_images = images.shape[0]
  if size is None:
    for sidel in range(int(np.floor(np.sqrt(nrof_images))),0,-1):
      if nrof_images % sidel == 0:
        size = [sidel, nrof_images//sidel]
        break
  assert nrof_images == size[0] * size[1]
  #images = np.maximum(images,0)
  #return imsave(inverse_transform(images), size, image_path)
  return imsave(images, size, image_path, out_dim = out_dim)

def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    print(images.shape)
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path, out_dim = None):
  image = np.squeeze(merge(images, size))
  # image.resize?
  if out_dim is not None:
    image = scipy.misc.imresize(image, out_dim)
  return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  #return np.array(cropped_image)/127.5 - 1.
  return np.array(cropped_image)/255.

def inverse_transform(images):
  #return (images+1.)/2.
  return images

def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  try:
    import moviepy.editor as mpy
  except:
    print("Error. Trying downloading ffmpeg")
    import imageio
    imageio.plugins.ffmpeg.download()
    import moviepy.editor as mpy
    
  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, option, generate_gif = False, save_input = False):
  bs = dcgan.args.batch_size
  image_frame_dim = int(math.ceil(dcgan.args.batch_size**.5))
  if option == 0: #normal samples
    sample_images = get_img(dcgan, 0, bs, dcgan.args.dataset_name, test=True)
    #z_sample = np.random.uniform(-0.5, 0.5, size=(bs, dcgan.args.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict=standard_sample_dict(dcgan, sample_images))
    save_images(samples, './samples/test_v1_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
  elif option == 1: #z[idx] influence
    sample_images = get_img(dcgan, 0, bs, dcgan.args.dataset_name, test=True)
    sample_tform_info = dcgan.Problem.create_tform_info(dcgan.args)
    sample_g_inputs0 = dcgan.Problem.transform(sample_images,sample_tform_info)
    sample_g_inputs = np.repeat([sample_g_inputs0[0]], bs, axis=0)
    values = np.arange(0, 1, 1./dcgan.args.batch_size)
    for idx in random.sample(range(dcgan.args.z_dim),4):
      print(" [*] %d" % idx)
      #z_sample = np.random.uniform(-1, 1, size=(dcgan.args.batch_size , dcgan.args.z_dim))
      z_sample0 = np.random.uniform(-1, 1, size=(1, dcgan.args.z_dim))
      z_sample = np.repeat(z_sample0, bs, axis=0)
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.g_inputs:sample_g_inputs})

      save_images(samples, './samples/test_v2_arange_%s.png' % (idx))
      
      if generate_gif:
        make_gif(samples, './samples/test_v2_gif_%s.gif' % (idx))
  elif option == 4: #gif merged different people
    sample_images = get_img(dcgan, 0, bs, dcgan.args.dataset_name, test=True)
    sample_tform_info = dcgan.Problem.create_tform_info(dcgan.args)
    sample_g_inputs0 = dcgan.Problem.transform(sample_images,sample_tform_info)
    sample_g_inputs = np.repeat([sample_g_inputs0[0]], bs, axis=0)
    
    image_set = []
    values = np.arange(0, 1, 1./dcgan.args.batch_size)

    sqrt_z_dim = int(np.floor(np.sqrt(dcgan.args.z_dim)))
    for idx in range(sqrt_z_dim**2):
      print(" [*] %d" % idx)
      z_sample = np.zeros([dcgan.args.batch_size, dcgan.args.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.g_inputs:sample_g_inputs}))
      #make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [sqrt_z_dim, sqrt_z_dim]) \
        for idx in [i for i in range(64)] + [i for i in range(63, -1, -1)]]
    make_gif(new_image_set, './samples/test_v4_gif_merged.gif', duration=8)
  elif option == 6: #Original together with merged.
    #Prints: merged version
    #   and merged next to original
    #   and inputs
    batch_size = dcgan.args.batch_size
    
    for idx in range(min(8,int(np.floor(1000/batch_size)))):
      print(" [*] %d" % idx)
      
      sample_images = get_img(dcgan, idx*batch_size, batch_size, dcgan.args.dataset_name, test=True)
      sample_tform_info = dcgan.Problem.create_tform_info(dcgan.args)
      sample_g_inputs = dcgan.Problem.transform(sample_images,sample_tform_info)
      s_g_in_save = dcgan.Problem.safe_format(sample_g_inputs)
      
      sample_z = np.random.uniform(-1, 1, size=(batch_size , dcgan.args.z_dim))
      
      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: sample_z, dcgan.g_inputs: sample_g_inputs})
      
      merged = dcgan.Problem.merge(samples, sample_g_inputs, sample_tform_info)
      
      save_multiple(2, [merged, sample_images], 'test_v6_merged_compare_%s' % (idx))
      #save_images(sample_g_inputs, './samples/test_v6_merged_samples_%s_inputs.png' % (idx))
      if save_input:
        save_images(dcgan.Problem.safe_format(sample_g_inputs), './samples/test_v6_merged_samples_%s_inputs.png' % (idx), out_dim=[256,256])
      #save_images(merged, './samples/test_v6_merged_samples_%s.png' % (idx))
      save_images(merged, './samples/test_v6_merged_samples_%s.png' % (idx), out_dim=[256,256])
      save_multiple(4, [sample_images, s_g_in_save, samples, merged],'test_v6_process_%s' % (idx))     
  elif option == 7: ##Save 4x4(6x6) image of merged samples (not coloured). 
    """batch_size = dcgan.args.batch_size
    
    for idx in xrange(min(8,int(np.floor(1000/batch_size)))):
      print(" [*] %d" % idx)
      
      sample_images = get_img(dcgan, idx*batch_size, batch_size, dcgan.args.dataset_name, test=True)
      sample_masks = create_masks(dcgan)
      sample_g_input = dcgan.tform(sample_images,sample_masks)
      
      sample_z = np.random.uniform(-1, 1, size=(batch_size , dcgan.args.z_dim))
      
      samples = sess.run(dcgan.sampler, \
            feed_dict={dcgan.z: sample_z, dcgan.g_input: sample_g_input, dcgan.inputs: sample_images, dcgan.mask: sample_masks })

      merged = dcgan.merge(samples, sample_g_input, sample_masks)
            
      if dcgan.args.dataset_name == 'mnist':
        merged_subset = merged[0:36]
        save_images(merged_subset, [6, 6, 3], './samples/test_v7_merged_samples_%s.png' % (idx))      
      else:
        merged_subset = merged[0:16]
        save_images(merged_subset, [4, 4, 3], './samples/test_v7_merged_samples_%s.png' % (idx))"""           
  elif option == 8: ##different values of z. Version to avoid batch normalization effect if this causes troubles"
    """batch_size = dcgan.args.batch_size
    length = int(np.sqrt(dcgan.args.batch_size))
    
    sample_inputs0, sample_img0, sample_labels0 = get_img(dcgan, 0, batch_size, dcgan.args.dataset_name, test=True)
    
    class_z = np.random.randint(2, size=dcgan.args.z_dim)
    values = np.linspace(-1., 1., num=length)
    z_values = np.empty((0,dcgan.args.z_dim))
      
    for i in range(length): #create z
      for j in range(length):
        z_values = np.append(z_values, [class_z * values[i] + (1-class_z) * values[j]], axis=0)
    
    shuff = np.zeros((0,batch_size)) #2nd column: permutations of 0:63
    for i in xrange(batch_size):
      x = np.arange(batch_size)
      random.shuffle(x)
      shuff = np.append(shuff, [x], axis=0).astype(int)
    
    all_samples = np.empty((batch_size,batch_size,dcgan.args.output_height,dcgan.args.output_width,dcgan.args.c_dim))
    
    for idx in xrange(batch_size): #over all noice variations.
      print(" [*] %d" % idx) #(old problem: Batch normalisation!!!!!!!)
    
      sample_inputs = sample_inputs0
      sample_labels = sample_labels0
      sample_img = sample_img0
      
      #Standard:
      #sample_z = np.repeat([z_values[idx]], dcgan.args.batch_size, axis=0)
      #print("first z_values:")
      #print(z_values[idx,1:12])
      #In case z gets caught in batch normalisation:
      #sample_z = np.random.uniform(-1, 1, size=(batch_size , dcgan.args.z_dim))
      #sample_z[0,:] = z_values[idx]
      
      sample_z = np.zeros((batch_size,dcgan.args.z_dim))
      for i in range(batch_size):
        z = z_values[shuff[i,idx]]        
        sample_z[i,:] = z        
      
      if dcgan.args.dataset_name == "mnist" and config.use_labels:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: sample_z, dcgan.img: sample_img, dcgan.y: sample_labels})     
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: sample_z, dcgan.img: sample_img })     
      
      for i in range(batch_size):
        all_samples[i,shuff[i,idx],:,:,:] = np.copy(samples[i])
      
    for idx in range(batch_size):
      
      samples = all_samples[idx,:,:,:,:]
      
      col_img = colour_samples(samples, dcgan.args.dataset_name, dcgan.args.img_height)
      
      save_images(col_img, [image_frame_dim, image_frame_dim, 3], './samples/test_v8_diffz_%s.png' % (idx))"""  
  elif option == 9: #different values of z.
    batch_size = dcgan.args.batch_size
    length = int(np.sqrt(dcgan.args.batch_size))
    
    sample_images0 = get_img(dcgan, 0, batch_size, dcgan.args.dataset_name, test=True)
    sample_tform_info0 = dcgan.Problem.create_tform_info(dcgan.args)
    sample_g_inputs0 = dcgan.Problem.transform(sample_images0,sample_tform_info0)
      
    class_z = np.random.randint(2, size=dcgan.args.z_dim)
    values = np.linspace(-1., 1., num=length)
    z_values = np.empty((0,dcgan.args.z_dim))
      
    for i in range(length): #create z
      for j in range(length):
        z_values = np.append(z_values, [class_z * values[i] + (1-class_z) * values[j]], axis=0)
    
    #for idx in range(batch_size): #over all noice variations.
    for idx in range(min(64,batch_size)):
      print(" [*] %d" % idx)
    
      sample_images = np.repeat([sample_images0[idx]], batch_size, axis=0)
      sample_g_inputs = np.repeat([sample_g_inputs0[idx]], batch_size, axis=0)
      #sample_tform_info = np.repeat([sample_tform_info0[idx]], batch_size, axis=0) 
      
      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_values, dcgan.g_inputs: sample_g_inputs})
            
      save_images(samples, './samples/test_v9_diffz_%s.png' % (idx))
      
      if idx < 8 and save_input:
        save_images(samples, './samples/test_v9_viz_%s.png' % (idx), out_dim=[256,256])
        save_images(dcgan.Problem.safe_format(sample_g_inputs), './samples/test_v9_viz_%s_inputs.png' % (idx), out_dim=[256,256])
  elif option == 10: #Take pictures from samples_progress and put them into one file.
    for i in range(8):
      prog_pics_base = glob(os.path.join('./samples_progress','part{:1d}'.format(i+1), '*.jpg'))    
    
      #prog_pics_base = glob(os.path.join('./samples_progress', '*.jpg'))
      imreadImg = imread(prog_pics_base[0])
      
      prog_pics = [
            get_image(prog_pic,
                      input_height=dcgan.args.output_height,
                      input_width=dcgan.args.output_height,
                      resize_height=dcgan.args.output_height,
                      resize_width=dcgan.args.output_width,
                      crop=dcgan.args.crop,
                      grayscale=dcgan.args.grayscale) for prog_pic in prog_pics_base]

      prog_pics_conv = np.array(prog_pics).astype(np.float32)   
      
      print(prog_pics_conv.shape)
      
      #out_pics = prog_pics_conv.reshape((64,prog_pics_conv.shape[1],prog_pics_conv.shape[2],:))
      out_pics = np.reshape(prog_pics_conv, (64,prog_pics_conv.shape[1],prog_pics_conv.shape[2],-1))
      print(out_pics.shape)
      #save_images(out_pics[1:2:], [1, 1], './samples_progress/progress1.png')
      
      
      #save_images(out_pics, [image_frame_dim, image_frame_dim], './samples_progress/progress{:1d}.png'.format(i+1))
      #save_images(out_pics, [8, 8], './samples_progress/progress{:1d}.png'.format(i+1))
      save_images(out_pics, './samples_progress/progress{:1d}.png'.format(i+1))      
  elif option == 11: #Save pictures centered and aligned in ./data_aligned
    """
    if True: #training data
      if not os.path.exists('data_aligned'):
        os.makedirs('data_aligned')
        
      nr_samples = len(dcgan.data)
      batch_size = dcgan.args.batch_size
      print(nr_samples)
      print(batch_size)
      
      batch_idxs = nr_samples // batch_size
      for idx in range(batch_idxs):
        sample_inputs, _, _ = get_img(dcgan, idx*batch_size, batch_size, dcgan.args.dataset_name, test=False)  
        for i in range(batch_size):
          pic_idx = idx*batch_size + i
          save_images(sample_inputs[i:i+1:],
                          './data_aligned/al{:06d}.jpg'.format(pic_idx+1))
        print("Done [%s] out of [%s]" % (idx,batch_idxs))
      '''                  
      sample_inputs, _, _ = get_img(dcgan, 0, nr_samples, dcgan.args.dataset_name, test=False)
      for pic_idx in range(nr_samples):
        save_images(sample_inputs[pic_idx:pic_idx+1:], [1,1],
                        './data_aligned/aligned{:03d}.jpg'.format(pic_idx+1))
      '''
    
    if True: #test data   
      if not os.path.exists('data_test_aligned'):
        os.makedirs('data_test_aligned')
      nr_samples = 1000    
      sample_inputs, _, _ = get_img(dcgan, 0, nr_samples, dcgan.args.dataset_name, test=True)
      for pic_idx in range(nr_samples):
        save_images(sample_inputs[pic_idx:pic_idx+1:],
                        './data_test_aligned/aligned{:03d}.jpg'.format(pic_idx+1)) """    
  #elif option == 12: 
                        
def standard_sample_dict(dcgan, sample_images):
    sample_tform_info = dcgan.Problem.create_tform_info(dcgan.args)
    sample_g_inputs = dcgan.Problem.transform(sample_images,sample_tform_info) 
    z_sample = np.random.uniform(-1, 1, size=(dcgan.args.batch_size, dcgan.args.z_dim))
    
    return {dcgan.z: z_sample, dcgan.g_inputs: sample_g_inputs}
        
def get_img(dcgan, start_idx, batch_size, dataset, test=True):
    
    if dataset == 'mnist' or dataset == "cifar10":
      if test:
        sample_images = dcgan.data_X_val[start_idx:(start_idx+batch_size)]
      else:
        sample_images = dcgan.data_X[start_idx:(start_idx+batch_size)]
    #elif dataset == "cifar10":
      
    else:
      if test:
        sample_files = dcgan.data_paths_val[start_idx:(start_idx+batch_size)]
      else:
        sample_files = dcgan.data_paths[start_idx:(start_idx+batch_size)]
      sample = [
          get_image(sample_file,
                    input_height=dcgan.args.input_height,
                    input_width=dcgan.args.input_width,
                    resize_height=dcgan.args.output_height,
                    resize_width=dcgan.args.output_width,
                    crop=dcgan.args.crop,
                    grayscale=dcgan.args.grayscale) for sample_file in sample_files]
      if (dcgan.args.grayscale):
        sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_images = np.array(sample).astype(np.float32)
      
      ##Shift to (0,1):
      #sample_inputs = (sample_inputs + 1) / 2 
   
    #return sample_inputs, sample_img, sample_labels
    return sample_images

### My functions:

def get_z_range(z_dim, batch_size):
    
  side_length = int(np.sqrt(batch_size))
  class_z = np.random.randint(2, size=z_dim)
  values = np.linspace(-1., 1., num=side_length)
  z_range = np.empty((0,z_dim))
    
  for i in range(side_length):
    for j in range(side_length):
      z_range = np.append(z_range, [class_z * values[i] + (1-class_z) * values[j]], axis=0)
        
  return z_range     

def save_multiple(nr, pictures, name):

  batch_size = pictures[0].shape[0]
  output = np.empty_like(pictures[0])
  
  nrof_i = batch_size // nr
  
  #suffices = ['a','b','c','d']
  suffices = string.ascii_lowercase
  
  for out_idx in range(nr):
    for ds_idx in range(nr):
      output[ds_idx::nr] = pictures[ds_idx][(out_idx*nrof_i):((out_idx+1)*nrof_i):]
    
    save_images(output, './samples/' + name + suffices[out_idx] + '.png' )
   
