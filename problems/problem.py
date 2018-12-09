
import tensorflow as tf
import numpy as np
    
name = "no problemo"
g_tform_info_placeholder = tf.placeholder(tf.float32, [None], name='g_transform_info')
    
def problem_loss(x_tformed, g_tformed):
  return tf.constant(0.)
  
def transform_tf(x, g_tform_info):
  return tf.zeros_like(x, dtype=tf.float32)

def transform(x, g_tform_info):
  return np.zeros_like(x, dtype=np.float32)
  
def create_tform_info(args):
  return [0]*args.batch_size

def safe_format(tformed):
  return np.clip(tformed,0,1)

def merge(g_output, x_tformed, g_tform_info):
  return g_output
  
  
