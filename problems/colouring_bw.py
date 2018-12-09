
import tensorflow as tf
import numpy as np

from problems.problem import *

name = "coulour b&w pictures"
      
g_tf_info_placeholder = tf.placeholder(tf.float32, [None], name='g_transform_info')
    
def problem_loss(x_tformed, g_tformed):
  return tf.reduce_mean(tf.abs(x_tformed-g_tformed),[1,2,3])

def transform_tf(x, g_tf_info):
  bw = tf.reduce_mean(x,axis=3, keepdims=True)
  return tf.concat([bw,bw,bw],3)

def transform(x, g_tf_info):
  bw = np.mean(x,axis=3, keepdims=True)
  return np.repeat(bw,3,axis=3)
  
def create_tform_info(args):
  return [0]*args.batch_size

def safe_format(tformed):
  return tformed
  
def merge(g_output, x_tformed, g_tform_info):
  merged = g_output
  for itr in range(3): #make result such that transform(result) ~ x_tformed.
    merged = g_output - transform(g_output, None) + x_tformed
    merged = np.clip(merged,0,1)
  return merged
  