import os
import sys
import scipy.misc
import numpy as np
import tensorflow as tf
import argparse
from datetime import datetime

import model
import utils

def main(args):
  
  if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
  if not os.path.exists(args.sample_dir):
    os.makedirs(args.sample_dir)
  if not os.path.exists('samples_progress'):
    os.makedirs('samples_progress')
  for i in range(8):
    if not os.path.exists('samples_progress/part{:1d}'.format(i+1)):
      os.makedirs('samples_progress/part{:1d}'.format(i+1))
  
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True 
  
  with open(args.settings_file_name,"a") as settings_file:
    for key, val in sorted(vars(args).items()):
        settings_file.write(key + ": " + str(val) + "\n")
  
  with open(args.progress_file_name,"a") as prog_file:
    prog_file.write("\n" + datetime.now().strftime("%H:%M:%S ") + "Started\n")
            
  with tf.Session(config=run_config) as sess:
    dcgan = model.DCGAN(sess, args)

    if args.train:
      dcgan.train()
      
      with open(args.progress_file_name,'a') as prog_file:
        prog_file.write("\n" + datetime.now().strftime("%H:%M:%S ") + "Finished training.\n")
    else:
      if not dcgan.load(args.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")

    # Below is codes for visualization
    if args.vis_type == 0:
      vis_options = [6,7,9,10]
      for option in vis_options:
        print("Visualizing option %s" % option)
        OPTION = option
        #utils.visualize(sess, dcgan, args, OPTION)
        utils.visualize(sess, dcgan, OPTION, save_input = True)
    else:
      OPTION = args.vis_type
      utils.visualize(sess, dcgan, OPTION)

      
def parse_arguments(argv):
  parser = argparse.ArgumentParser()
    
  parser.add_argument("--nrof_epochs", type=int,
    help="Epochs to train [8]", default=8)
  parser.add_argument("--learning_rate", type=float,
    help="Learning rate of for adam [0.0002]", default=0.0002) 
  parser.add_argument("--beta1", type=float,
    help="Momentum term of adam [0.5]", default=0.5)
  parser.add_argument("--train_size", type=int,
    help="Number of train images to be used. If None, uses all. [None]", default=None)
  parser.add_argument("--batch_size", type=int,
    help="The size of batch images [64]", default=64)
  parser.add_argument("--input_height", type=int,
    help="The size of image to use (will be center cropped). [108]", default=108)
  parser.add_argument("--input_width", type=int,
    help="The size of image to use (will be center cropped). If None, same value as input_height [None]", default=None)
  parser.add_argument("--output_height", type=int,
    help="The size of the output images to produce [64]", default=64)
  parser.add_argument("--output_width", type=int,
    help="The size of the output images to produce. If None, same value as output_height [None]", default=None)
  parser.add_argument("--dataset_name", type=str,
    help="The name of dataset [celebA, mnist, lsun]", default="celebA")
  parser.add_argument("--input_fname_pattern", type=str,
    help="Glob pattern of filename of input images [*]", default="*.jpg")
  parser.add_argument("--sample_dir", type=str,
    help="Directory name to save the image samples [samples]", default="samples")
  parser.add_argument("--checkpoint_dir", type=str,
    help="Directory name to save the checkpoints [checkpoint]", default="checkpoint")
  parser.add_argument("--train",
    help="True for training, False for testing [False]", action='store_true')
  parser.add_argument("--crop",
    help="True for training, False for testing [False]", action='store_true')
  
  parser.add_argument("--vis_type", type=int,
    help="Visualization option; 0=all. [0]", default=0)
  parser.add_argument("--lambda_loss", type=float,
    help="Coefficient of additional loss. [10.]", default=10.)
  parser.add_argument("--z_dim", type=int,
    help="Dimension of the random input. [100]", default=100)
  parser.add_argument("--g_feature_dim", type=int,
    help="Dimension of the bottleneck layer. [100]", default=100)
  parser.add_argument("--max_reach", type=int,
    help="Parameter for mask creation. [12]", default=12)

  parser.add_argument("--data_dir", type=str,
    help="Directory name to load data. [data]", default="../../../data")

  parser.add_argument('--settings_file_name', type=str,
    help='Name (path) of the settings file.', default='settings.txt')
  parser.add_argument('--progress_file_name', type=str,
    help='Name (path) of the progress file.', default='progress.txt')    
  parser.add_argument('--problem_name', type=str,
    help='Name (path) of the problem python file.', default='problems.problem')
  parser.add_argument('--save_freq', type=int,
    help='How often picuteres are saved.', default=100)
    
  # Output Args
  args = parser.parse_args(argv)
  
  # Change some defaults
  if args.dataset_name == "mnist":
    args.input_height = 28
    args.output_height = 28
  if args.dataset_name == "cifar10":
    args.input_height = 32
    args.output_height = 32
  if args.input_width is None:
    args.input_width = args.input_height
  if args.output_width is None:
    args.output_width = args.output_height    
    
    
  options = vars(args)
  
  with open(args.settings_file_name,"w") as settings_file:
      settings_file.write("\n" + " ".join(sys.argv) + "\n\n")
            
  return args
 
if __name__ == '__main__':
  args = parse_arguments(sys.argv[1:])
  main(args)
  
  