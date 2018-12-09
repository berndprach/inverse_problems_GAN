
import tensorflow as tf
import numpy as np

#from problems.problem import *

name = "flexible inpainting"
      
#g_tf_info_placeholder = tf.placeholder(tf.float32, name='g_transform_info')
g_tform_info_placeholder = tf.placeholder(tf.float32, name='g_transform_info')
    
def problem_loss(x_tformed, g_tformed):
  return tf.reduce_mean(tf.abs(x_tformed-g_tformed),[1,2,3])
  
def transform_tf(x, g_tf_info):
  return g_tf_info * (x + 1) - 1

def transform(x, g_tf_info):
  return g_tf_info * (x + 1) - 1
  
def create_tform_info(args):
  # create masks. (mask[i,j] = 1{pixel (i,j) is known})
  batch_of_masks = np.zeros([args.batch_size, args.output_height, args.output_width, args.c_dim])
  max_reach = args.max_reach
  for idx in range(args.batch_size):
    bs = args.batch_size
    h = args.output_height + 2*max_reach
    w = args.output_width + 2*max_reach
    p = 4 * max_reach / h / w
    center_val = np.random.randint(1, max_reach, size=(h, w))
    mask_ = np.ones([h, w]).astype(int)
    is_center = np.random.binomial(1, p, size=(h, w))

    for i in range(h):
        for j in range(w):
            if is_center[i,j] == 0: continue
            for d_i in range(-center_val[i,j],center_val[i,j]+1):
                if i + d_i < 0: continue
                if i + d_i >= h: continue
                for d_j in range(-center_val[i,j],center_val[i,j]+1):
                    if j + d_j < 0: continue
                    if j + d_j >= w: continue
                    if d_i**2 + d_j**2 <= center_val[i,j]**2:
                        mask_[i+d_i,j+d_j] = 0
   
    mask = np.repeat(mask_[max_reach:-max_reach,max_reach:-max_reach, np.newaxis],args.c_dim,axis=2)  
    batch_of_masks[idx] = mask
   
  return batch_of_masks

def safe_format(tformed):
  return np.maximum(tformed,0)
  
def merge(g_output, x_tformed, g_tform_info):
  return g_tform_info * x_tformed + (1-g_tform_info) * g_output
  