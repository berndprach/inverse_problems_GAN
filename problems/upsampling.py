
import tensorflow as tf
import numpy as np

from problems.problem import *

name = "upsample pictures"
      
g_tf_info_placeholder = tf.placeholder(tf.float32, [None], name='g_transform_info')
    
def problem_loss(x_tformed, g_tformed):
  return tf.reduce_mean(tf.abs(x_tformed-g_tformed),[1,2,3])

def transform_tf(x, g_tf_info): #use assign?
  h, w = x.shape[1:3]
  assert h % 8 == 0 and w % 8 == 0
  h0, w0 = h//8, w//8
  avg_line_list = []
  for i in range(8):
    avg_list = []
    for j in range(8):
      region = x[:,i*h0:(i+1)*h0,j*w0:(j+1)*w0,:]
      avg_list += [tf.reduce_mean(region,axis=[1,2])]*8
    avg_line_list += [tf.stack(avg_list,axis=1)]*8
  output = tf.stack(avg_line_list,axis=1)
  return output

def transform(x, g_tf_info):
  output = np.zeros_like(x, dtype=np.float32)
  h, w = output.shape[1:3]
  assert h % 8 == 0 and w % 8 == 0
  h0, w0 = h//8, w//8
  for i in range(8):
    for j in range(8):
      region = x[:,i*h0:(i+1)*h0,j*w0:(j+1)*w0,:]
      avg = np.mean(region,axis=(1,2), keepdims=True)
      avg_region = np.tile(avg,[1,8,8,1])
      output[:,i*h0:(i+1)*h0,j*w0:(j+1)*w0,:] = avg_region
  return output
  
def create_tform_info(args):
  return [0]*args.batch_size

def safe_format(tformed):
  return tformed
  
def merge(g_output, x_tformed, g_tform_info):
  merged = np.copy(g_output)
  for itr in range(3): #make result such that transform(result) ~ x_tformed.
    merged = g_output - transform(g_output, None) + x_tformed
    merged = np.clip(merged,0,1)
  return merged

  