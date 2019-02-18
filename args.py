from __future__ import print_function

import os

import tensorflow as tf

import argparse

def format_parser(parser):
    """Build the arguements parser"""

    # Hyperparameters regarding the neural network
    parser.add_argument("--checkpoint_dir", type=str, default='', help="Filepath to directory where checkpoints are stored")
    parser.add_argument("--rl_dir", type=str, default='', help="Filepath to directory where policies are stored")
    parser.add_argument("--initialize_rl", type=str, default='n', help="If yes, policy is initialized using newest seq2seq model")
    parser.add_argument("--continue_training", type=str, default='y', help="If no, initializes a brand new model for training")
    parser.add_argument("--train_file", type=str, default='', help="Filepath to training data")
    parser.add_argument("--model_file", type=str, default=None, help="Filepath to saved model to continue training")
    parser.add_argument("--vocab_file", type=str, default=None, help="Filepath to precomputed vocab file")
    parser.add_argument("--encoder_file", type=str, default=None, help="Filepath to saved encoder model for inference")
    parser.add_argument("--decoder_file", type=str, default=None, help="Filepath to saved decoder model for inference")
    parser.add_argument("--init_sentences", type=str, default=None, help="Filepath to saved starting sentences")

    parser.add_argument("--embedding_dim", type=int, default=128, help="Size of word embedding vectors")
    parser.add_argument("--units", type=int, default=256, help="Size of network layers")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_examples", type=int, default=324401, help="Number of dataset pairs to train on")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of dataset to reserve for testing")
