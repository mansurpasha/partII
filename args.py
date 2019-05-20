from __future__ import print_function

import os

import tensorflow as tf

import argparse

def format_parser(parser):
    """Build the arguements parser"""

    # checkpoint filepaths, initialize models or continue training
    parser.add_argument("--checkpoint_dir", type=str, default='', help="Filepath to directory where checkpoints are stored")
    parser.add_argument("--tensorboard_dir", type=str, default='', help="Filepath to directory where Tensorboard writer results are stored")
    parser.add_argument("--rl_dir", type=str, default='', help="Filepath to directory where policies are stored")
    parser.add_argument("--initialize_rl", type=str, default='n', help="(y/n) If yes, policy is initialized using newest seq2seq model")
    parser.add_argument("--continue_training", type=str, default='y', help="(y/n) If no, initializes a brand new model for training")

    # filepath arguments
    parser.add_argument("--train_file", type=str, default='', help="Filepath to training data")
    parser.add_argument("--length_file", type=str, default='', help="Filepath to lengths of data")
    parser.add_argument("--model_file", type=str, default=None, help="Filepath to saved model to continue training")
    parser.add_argument("--vocab_file", type=str, default=None, help="Filepath to precomputed vocab file")
    parser.add_argument("--encoder_file", type=str, default=None, help="Filepath to saved encoder model for inference")
    parser.add_argument("--decoder_file", type=str, default=None, help="Filepath to saved decoder model for inference")
    parser.add_argument("--init_sentences", type=str, default=None, help="Filepath to saved starting sentences")

    parser.add_argument("--eval_file", type=str, default=None, help="Filepath to evaluation testing file")
    parser.add_argument("--old_scope", type=str, default=None, help="Name of variable scope of model to be evaluated")

    # model hyperparameters
    parser.add_argument("--model_name", type=str, default="S2S_placeholder", help="Top level variable scope of Seq2Seq model trained")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Size of word embedding vectors")
    parser.add_argument("--units", type=int, default=256, help="Size of network layers")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_examples", type=int, default=324401, help="Number of dataset pairs to train on")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of dataset to reserve for testing")
    parser.add_argument("--model_type", type=str, default=None, help="prev2, forward, or backward, depending on model being trained")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate for optimizer")
    parser.add_argument("--keep_prob", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--display_step", type=int, default=300, help="Number of steps between displaying model performance")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in neural network")
    parser.add_argument("--max_target_length", type=int, default=40, help="Maximum permitted length of output sentence")

    # model variants
    parser.add_argument("--use_dropout", type=str, default='y', help="(y/n) If yes, uses Dropout when training models")
    parser.add_argument("--use_attention", type=str, default='n', help="(y/n) If yes, adds an attention layer to the encoder")
    parser.add_argument("--attention_state", type=int, default=5, help="Number of states in the attention mechanism")
    parser.add_argument("--beam_search", type=str, default='n', help="(y/n) If yes, uses a beam_search_decoder")
    parser.add_argument("--beam_width", type=int, default=5, help="Number of candidate responses tracked by beam decoder")

    # reinforcement learning parameters
    parser.add_argument("--gamma", type=float, default='0.99', help="Discount coefficient when discounting batch rewards")

