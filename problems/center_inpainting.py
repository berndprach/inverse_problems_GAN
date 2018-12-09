
import tensorflow as tf
import numpy as np

from problems.problem import *

name = "center inpainting"
      
g_tf_info_placeholder = tf.placeholder(tf.float32, [None], name='g_transform_info')
    
def problem_loss(x_tformed, g_tformed):
  return tf.reduce_mean(tf.abs(x_tformed-g_tformed),[1,2,3])

def merge(g_output, x_tformed, g_tform_info):
  h, w = x_tformed.shape[1:3]
  h4, w4 = h//6, w//6
  merged = np.copy(x_tformed)
  merged[:,h4:h-h4,w4:w-w4,:] = g_output[:,h4:h-h4,w4:w-w4,:]
  return merged

def transform_tf(x, g_tf_info):
  not_x = - tf.ones_like(x, dtype=tf.float32)
  mask = np.ones(x.get_shape(), dtype=np.float32)
  mask0 = np.zeros(x.get_shape(), dtype=np.float32)
  mask = merge(mask0, mask, None)
  output = mask * x + (1-mask) * not_x
  return output

  
def transform(x, g_tf_info):
  not_x = - np.ones_like(x, dtype=np.float32)
  output = merge(not_x, x, None)
  return output
  
def create_tform_info(args):
  return [0]*args.batch_size

def safe_format(tformed):
  return np.clip(tformed,0,1)
  
  