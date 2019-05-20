import processing
import models
from nltk.tokenize import word_tokenize

import tensorflow as tf
from tensorflow.train import init_from_checkpoint

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import time
import argparse
import random
import pickle
from tensorflow.python import debug as tf_debug

from processing import load_preprocess
from models import RLModel, RLModel
from processing import LanguageIndex
import args
import utils


import math

parser = argparse.ArgumentParser()
args.format_parser(parser)

parameters, unparsed = parser.parse_known_args()

(encoder_input, decoder_input, decoder_output) = load_preprocess(parameters.train_file)
#(encoder_lengths, decoder_lengths, decoder_lengths2) = load_preprocess(parameters.length_file)
vocab = processing.LanguageIndex

vocab = processing.load_preprocess(parameters.vocab_file)


init_sentences = open("data/self_dialogue_corpus/processed/init_sentences", 'r').readlines()

# Given a seq2seq model, an input, and a desired outut, calculate the probability of the model producing this result
# Format of arguments:
def calculate_probability(model, input, expected_response, lang):
    expected_response = utils.sentence_to_idx(expected_response, lang)
    expected_response = expected_response.tolist()
    expected_response.append(lang.word2idx["<end>"])
    expected_response = np.array(expected_response)

    running_probability = 1

    sess = tf.Session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # Training_output is a list of logit probability distributions
    logits = sess.run(model.training_output[0], {model.inputs_: input,
                                                 model.targets_: expected_response,
                                                 model.target_lengths_: [len(expected_response)],
                                                 model.max_target_length_: len(expected_response)})

    for logit,word in zip(logits,expected_response):
        probability = logit[word]
        running_probability *= probability

    return running_probability

# Calculates maximum mutual information rewards for the MI model
# Accepts a prev2 model and a backwards model
class MIRewardCalculator():
    def __init__(self, prev2, backward):
        self.prev2_model = prev2
        self.backward = backward

    # Calculate pairwise mutual information
    # Accepts two index sequences as arguments
    # Returns log prob
    def calculate_r1(self, action, state_p, state_q, lang):
        forward_likelihood = (1 / len(action)) * math.log10(calculate_probability(self.prev2_model,
                                                                                  np.concatenate((state_p, state_q)),
                                                                                  utils.idx_to_sentence(action, lang),
                                                                                  lang))
        backward_likelihood = (1 / len(state_q)) * math.log10(calculate_probability(self.backward_model,
                                                                                    action,
                                                                                    utils.idx_to_sentence(state_q,
                                                                                                          lang),
                                                                                    lang))
        return forward_likelihood + backward_likelihood


# Calculates rewards for the RL model making use of 3 baseline seq2seq models to perform calculations
class RewardCalculator():
    def __init__(self, forward, backward, prev2, dull_responses):
        self.dull_responses = dull_responses
        self.prev2_model = prev2
        self.forward_model = forward
        self.backward_model = backward

    # calculates the average negative log likelihood of an utterance being responded to with a dull response
    # input: utterance as int sequence, language dictionary
    # output: negative log prob
    def calculate_r1(self, utterance, lang):
        cumulative_likelihood = 0
        for s in self.dull_responses:
            # normalized by length of sentence in tokens, might want to store boring sentences as int sequences
            cumulative_likelihood += ((1 / len(utils.sentence_to_idx(s, lang))) *
                                      math.log10(calculate_probability(self.forward_model, utterance, s,
                                                                       lang)))
        return -1 * (1 / len(self.dull_responses)) * cumulative_likelihood

    # calculates the negative log of the cosine of similarity between two consecutive turns of dialogue from the policy
    # requires turns be submitted as the encoded form, i.e. the hidden state when finished encoding
    def calculate_r2(t1, t2):
        return -math.log10(np.dot(t1, t2) / (len(t1) * len(t2)))

    # calculate the log likelihood of the seq2seq generating utterance a based on the two sentence state [p,q]
    # plus the log likelihood of the backwards_seq2seq generating (q|a)
    # input: action, state_p, and state_q should all be index sequences
    def calculate_r3(self, action, state_p, state_q, lang):
        forward_likelihood = (1 / len(action)) * math.log10(calculate_probability(self.prev2_model,
                                                                                  np.concatenate((state_p, state_q)),
                                                                                  utils.idx_to_sentence(action, lang),
                                                                                  lang))
        backward_likelihood = (1 / len(state_q)) * math.log10(calculate_probability(self.backward_model,
                                                                                    action,
                                                                                    utils.idx_to_sentence(state_q,lang),
                                                                                    lang))
        return forward_likelihood + backward_likelihood

# Class that initializes simulated conversations, advances them one turn at a time, and tracks current state
class Simulation():
    def __init__(self, policy, params, lang, max_len, calculator):
        self.policy = policy
        self.lang = lang
        self.max_len = max_len
        self.calculator = calculator
        # Max length of conversations in turns
        self.sequence_limit = 10
        # store turns of dialogue as index arrays
        self.utterances = []
        # store turns of dialogue as sentences
        self.conversation = []
        self.done = False

    def sentence2idx(self, sentence):
        return utils.sentence_to_idx(sentence, self.lang)

    # Initialises simulation
    def reset(self, start_sentence):
        # Wipe stored conversation
        self.utterances = []
        # Initialize with sentence from initialization dataset
        self.utterances.append(self.sentence2idx(start_sentence))
        self.done = False

    # Steps through the simulation one turn, generating a new response and updating states as necessary
    # Returns the reward for this turn
    def step(self):
        # create input for prev2 model
        inputs = np.concatenate((self.utterances[-2], self.utterances[-1]))

        sess = tf.Session()
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # Training_output is a list of logit probability distributions
        output = sess.run(self.policy.inference_output, {self.policy.inputs_: input,
                                                         self.policy.max_target_length_: self.max_len})


        # output[1] is argmax of logits, so returns sequence picked
        observation = output[1]

        # store in simulation
        self.utterances.append(observation)

        # if max conversation length reached, set done flag to true
        if len(self.utterances) > self.sequence_limit:
            self.done = True
        reward = self.calculate_reward
        return reward

    def calculate_reward(self):
        r1 = self.calculator.calculate_r1(self.utterances[-1], self.lang)

        state_p = self.utterances[-3]
        state_q = self.utterances[-2]

        sess = tf.Session()
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # Get encoder hidden state after utterance is input (sentence encoding)
        encoded_states = sess.run(self.policy.enc_state, {self.policy.inputs_: [state_p, state_q]})


        # r2 = self.calculator.calculate_r2(p_hidden, q_hidden)
        r2 = self.calculator.calculate_r2(encoded_states[0], encoded_states[1])

        r3 = self.calculator.calculate_r3(self.utterances[-1], self.utterances[-3], self.utterances[-2], self.lang)

        a1, a2, a3 = 0.25, 0.25, 0.5

        return a1 * r1 + a2 * r2 + a3 * r3


# Reset the graph
tf.reset_default_graph()

# Instantiate the RLModel
RLModel = RLModel(params=parameters, name="Self_Mutual_Info", language=vocab)

# Initialize Session
sess = tf.Session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver(max_to_keep=5)
#saver.restore(sess, parameters.checkpoint_dir)

# Seq2Seq models all saved with the same variable scope, we can use this to target variables in the new
# RL model, specifically don't want to learn the optimizer settings from the old model
# as we want initial training and exploring of the action space to move quickly
init_from_checkpoint(parameters.checkpoint_dir,
  {'Seq2SeqPolicy/decoder/dense/bias': 'Self_Mutual_Info/decoder/dense/bias',
  'Seq2SeqPolicy/decoder/dense/kernel': 'Self_Mutual_Info/decoder/dense/kernel',
  'Seq2SeqPolicy/decoder/lstm_cell/bias': 'Self_Mutual_Info/decoder/lstm_cell/bias',
  'Seq2SeqPolicy/decoder/lstm_cell/kernel': 'Self_Mutual_Info/decoder/lstm_cell/kernel',
  'Seq2SeqPolicy/embeddings': 'Self_Mutual_Info/embeddings',
  'Seq2SeqPolicy/rnn/encoder_lstm/bias': 'Self_Mutual_Info/rnn/encoder_lstm/bias',
  'Seq2SeqPolicy/rnn/encoder_lstm/kernel': 'Self_Mutual_Info/rnn/encoder_lstm/kernel'})

# Initialise 3 new seq2seq models, renaming the inherited variables from file
Prev2Model = RLModel(params=parameters, name="Prev2Model", language=vocab)
init_from_checkpoint(parameters.checkpoint_dir, {'Seq2SeqPolicy/': 'Prev2ModelS2S'})
ForwardModel = RLModel(params=parameters, name="ForwardModel", language=vocab)
init_from_checkpoint(parameters.checkpoint_dir, {'Seq2SeqPolicy/': 'ForwardModelS2S'})
BackwardModel = RLModel(params=parameters, name="BackwardModel", language=vocab)
init_from_checkpoint(parameters.checkpoint_dir, {'Seq2SeqPolicy/': 'BackwardModelS2S'})

# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/pg/test")

## Losses
tf.summary.scalar("Loss", RLModel.loss)

## Reward mean
tf.summary.scalar("Reward_mean", RLModel.mean_reward_ )

write_op = tf.summary.merge_all()

calc = RewardCalculator(Prev2Model, ForwardModel, BackwardModel, ["i don't know", "meh", "huh"])
sim = Simulation(RLModel, parameters, vocab, parameters.max_target_length, calc)

# calc = RewardCalculator(encoder, decoder, encoder, decoder, encoder, decoder, ["i don't know", "meh", "huh"])
# sim = Simulation(encoder, decoder, language, max_length, calc)

def make_batch(batch_size):
    # Initialize lists: states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards
    states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []

    # Reward of batch is also a trick to keep track of how many timestep we made.
    # We use to to verify at the end of each episode if > batch_size or not.

    # Keep track of how many episodes in our batch (useful when we'll need to calculate the average reward per episode)
    episode_num = 1

    # Get a new state
    #state = game.get_state().screen_buffer
    state = random.choice(init_sentences)
    #state, stacked_frames = stack_frames(stacked_frames, state, True)
    sim.reset(state)

    while True:
        # Run State Through Policy & Calculate Action
        reward = sim.step()
        done = sim.done

        # Store results
        states.append(np.concatenate((sim.utterances[-3],sim.utterances[-2])))
        actions.append(sim.utterances[-1])
        rewards_of_episode.append(reward)

        if done:
            # Append the rewards_of_batch to reward_of_episode
            rewards_of_batch.append(rewards_of_episode)

            # Calculate gamma Gt
            discounted_rewards.append(discount_and_normalize_rewards(rewards_of_episode))

            # If the number of rewards_of_batch > batch_size stop the minibatch creation
            # (Because we have sufficient number of episode mb)
            # Remember that we put this condition here, because we want entire episode (Monte Carlo)
            # so we can't check that condition for each step but only if an episode is finished
            if len(np.concatenate(rewards_of_batch)) > batch_size:
                break

            # Reset the transition stores
            rewards_of_episode = []

            # Add episode
            episode_num += 1

            # Start a new episode
            state = random.choice(init_sentences)
            sim.reset(state)

    return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewards_of_batch), np.concatenate(
        discounted_rewards), episode_num

def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * parameters.gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards

allRewards = []

total_rewards = 0
maximumRewardRecorded = 0
mean_reward_total = []
epoch = 1
average_reward = []

while epoch < parameters.num_epochs + 1:

    # Gather training data
    states_mb, actions_mb, rewards_of_batch, discounted_rewards_mb, nb_episodes_mb = make_batch(parameters.batch_size)

    # Backprop
    loss_, _ = sess.run([RLModel.loss, RLModel.train_op],
                        feed_dict={RLModel.inputs_: states_mb,
                                   RLModel.actions_: actions_mb,
                                   RLModel.discounted_episode_rewards_: discounted_rewards_mb
                                   })

    print("Training Loss: {}".format(loss_))

    ### These part is used for analytics
    # Calculate the total reward ot the batch
    total_reward_of_that_batch = np.sum(rewards_of_batch)
    allRewards.append(total_reward_of_that_batch)

    # Calculate the mean reward of the batch
    # Total rewards of batch / nb episodes in that batch
    mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, nb_episodes_mb)
    mean_reward_total.append(mean_reward_of_that_batch)

    # Calculate the average reward of all training
    # mean_reward_of_that_batch / epoch
    average_reward_of_all_training = np.divide(np.sum(mean_reward_total), epoch)

    # Calculate maximum reward recorded
    maximumRewardRecorded = np.amax(allRewards)

    print("==========================================")
    print("Epoch: ", epoch, "/", parameters.num_epochs)
    print("-----------")
    print("Number of training episodes: {}".format(nb_episodes_mb))
    print("Total reward: {}".format(total_reward_of_that_batch, nb_episodes_mb))
    print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
    print("Average Reward of all training: {}".format(average_reward_of_all_training))
    print("Max reward for a batch so far: {}".format(maximumRewardRecorded))

    # Write TF Summaries
    summary = sess.run(write_op, feed_dict={RLModel.inputs_: states_mb,
                                            RLModel.actions_: actions_mb,
                                            RLModel.discounted_episode_rewards_: discounted_rewards_mb,
                                            RLModel.mean_reward_: mean_reward_of_that_batch
                                            })

    # summary = sess.run(write_op, feed_dict={x: s_.reshape(len(s_),84,84,1), y:a_, d_r: d_r_, r: r_, n: n_})
    writer.add_summary(summary, epoch)
    writer.flush()

    # Save Model
    if epoch % 10 == 0:
        saver.save(sess, "checkpoints/rl_checkpoints/mutual.ckpt")
        print("Model saved")
    epoch += 1

