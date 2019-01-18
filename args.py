from __future__ import print_function

import os

import tensorflow as tf

import argparse

def format_parser(parser):
    """Build the arguements parser"""
    # Register the boolean type. This allows us to use booleans as arguments
    parser.register('type', 'bool', lambda v: v.lower == "true")

    # Hyperparameters regarding the neural network
    parser.add_argument("--checkpoint_dir", type=str, default='', help="Location of checkpoint dir")
    parser.add_argument("--train_file", type=str, default='', help="Filepath to training data")