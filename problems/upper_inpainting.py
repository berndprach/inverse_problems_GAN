
import tensorflow as tf
import numpy as np

from problems.problem import *

name = "center inpainting"
      
g_tf_info_placeholder = tf.placeholder(tf.float32, [None], name='g_transform_info')

def problem_loss(x_tformed, g_tformed):
  return tf.reduce_mean(tf.abs(x_tformed-g_tformed),[1,2,3])

#def merge_images(g_output, x_tformed, g_tform_info):
def merge_images(g_output, x_tformed):
  h, w = x_tformed.shape[1:3]
  h0 = 4*h//10
  merged = np.copy(x_tformed)
  merged[:,:h-h0,:,:] = g_output[:,:h-h0,:,:]
  return merged

def transform_tf(x, g_tf_info):
  not_x = - tf.ones_like(x, dtype=tf.float32)
  mask = np.ones(x.get_shape(), dtype=np.float32)
  mask0 = np.zeros(x.get_shape(), dtype=np.float32)
  mask = merge_images(mask0, mask)
  output = mask * x + (1-mask) * not_x
  return output
  
def transform(x, g_tf_info):
  not_x = - np.ones_like(x, dtype=np.float32)
  output = merge_images(not_x, x)
  return output
  
def create_tform_info(args):
  return [0]*args.batch_size

#def safe_format(tformed):
#  return np.clip(tformed,0,1)
  
def merge(g_output, x_tformed, g_tform_info): # (with colouring)
  if x_tformed.shape[3] != 1:
    return merge_images(g_output, x_tformed, g_tform_info)
  else:
    x0 = np.zeros_like(x_tformed, dtype=np.float32)
    #x_tformed_col = np.concatenate([x_tformed, x0, x0], axis=3)
    x_tformed_col = np.concatenate([x0, x0, x_tformed], axis=3)
    g_output_col = np.concatenate([g_output, g_output, g_output], axis=3)
    return merge_images(g_output_col, x_tformed_col)
  
  