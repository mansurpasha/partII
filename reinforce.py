import processing
import models
from nltk.tokenize import word_tokenize

from __future__ import absolute_import, division, print_function

import tensorflow as tf

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

from processing import load_preprocess
from models import Seq2Seq
import args
import utils

parser = argparse.ArgumentParser()
args.format_parser(parser)

parameters, unparsed = parser.parse_known_args()

(encoder_input, decoder_input, decoder_output) = load_preprocess(parameters.train_file)
#(encoder_lengths, decoder_lengths, decoder_lengths2) = load_preprocess(parameters.length_file)
vocab = processing.LanguageIndex
vocab = pickle.load(parameters.vocab_file)
import math

def calculate_probability(model, input, expected_response, lang):
    expected_response = utils.sentence_to_idx(expected_response, lang)
    expected_response = expected_response.tolist()
    expected_response.append(lang.word2idx["<end>"])
    expected_response = np.array(expected_response)

    running_probability = 1

    sess = tf.Session()
    logits = sess.run(model.training_output[0], {model.inputs_: input,
                                        model.targets_: expected_response,
                                        model.target_lengths_: len(expected_response)})

    for logit,word in zip(logits,expected_response):
        probability = logit[word]
        running_probability *= probability

    return running_probability

class RewardCalculator():
    def __init__(self, encoder_2state, decoder_2state, seq2seq_encoder, seq2seq_decoder, seq2seq_back_encoder,
                 seq2seq_back_decoder, dull_responses):
        self.double_enc = encoder_2state
        self.double_dec = decoder_2state
        self.forward_enc = seq2seq_encoder
        self.forward_dec = seq2seq_decoder
        self.back_enc = seq2seq_back_encoder
        self.back_dec = seq2seq_back_decoder

        self.dull_responses = dull_responses
        self.prev2_model = Seq2Seq(name="S2S_prev2")
        self.forward_model = Seq2Seq(name="S2S_forward")
        self.backward_model = Seq2Seq(name="S2S_backward")


    # calculates the average negative log likelihood of an utterance being responded to with a dull response
    # input: utterance should be a padded max_length
    def calculate_r1(self, utterance, lang):
        cumulative_likelihood = 0
        for s in self.dull_responses:
            cumulative_likelihood += ((1 / len(sentence_to_idx_unpadded(s, lang))) *
                                      math.log10(calculate_probability(self.forward_enc, self.forward_dec, utterance, s,
                                                                       lang)))
        return -1 * (1 / len(dull_responses)) * cumulative_likelihood

    # calculates the negative log of the cosine of similarity between two consecutive turns of dialogue from the policy
    # requires turns be submitted as the encoded form, i.e. the hidden state when finised encoding
    def calculate_r2(t1, t2):
        return -math.log10(np.dot(t1, t2) / (len(t1) * len(t2)))

    # calculate the log likelihood of the seq2seq generating utterance a based on the two sentence state [p,q]
    # plus the log likelihood of the backwards_seq2seq generating (q|a)
    # input: action, state_p, and state_q should all be padded max_length index sequences
    def calculate_r3(action, state_p, state_q, lang):
        forward_likelihood = (1 / len(action)) * math.log10(calculate_probability(self.double_enc,
                                                                                  self.double_dec,
                                                                                  np.concatenate((state_p, state_q)),
                                                                                  processing.idx2sentence(action, lang),
                                                                                  lang))
        backward_likelihood = (1 / len(state_q)) * math.log10(calculate_probability(self.back_enc,
                                                                                    self.back_dec,
                                                                                    action,
                                                                                    processing.idx2sentence(state_q,
                                                                                                            lang),
                                                                                    lang))
        return forward_likelihood + backward_likelihood


class Simulation():
    def __init__(self, encoder, decoder, lang, max_len, calculator):
        self.encoder = encoder
        self.decoder = decoder
        self.lang = lang
        self.max_len = max_len
        self.calculator = calculator
        self.sequence_limit = 10
        # store turns of dialogue as padded index arrays
        self.utterances = []
        # store turns of dialogue as sentences
        self.conversation = []
        self.done = False

    def sentence2idx(self, sentence):
        return processing.sentence_to_idx(sentence, self.lang, self.max_len)

    def reset(self, start_sentence):
        self.utterances = []
        self.utterances.append(self.sentence2idx("start_of_conversation_token"))
        self.utterances.append(self.sentence2idx(start_sentence))
        self.done = False

    def step(self):
        inputs = np.concatenate((self.utterances[-2], self.utterances[-1]))
        inputs = tf.expand_dims(inputs, 0)

        hidden = [tf.zeros((1, self.encoder.enc_units))]
        enc_output, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.lang.word2idx['<start>']], 0)

        observation = ''

        for t in range(max_length):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_output)

            predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()
            turn.append(predicted_id)
            observation += self.lang.idx2word[predicted_id] + ' '

            if self.lang.idx2word[predicted_id] == '<end>':
                break

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        self.conversation.append(observation)
        self.utterances.append(self.sentence2idx(observation))
        if len(self.utterances) > 10:
            self.done = True
        reward = self.calculate_reward
        return reward

    def calculate_reward(self):
        r1 = self.calculator.calculate_r1(self.utterances[-1], self.lang)

        hidden = [tf.zeros((1, units))]
        state_p = self.utterances[-3]
        state_q = self.utterances[-2]

        inputs = tf.convert_to_tensor(state_p)
        inputs = tf.expand_dims(inputs, 0)
        _, p_hidden = encoder(inputs, hidden)
        inputs = tf.convert_to_tensor(state_q)
        inputs = tf.expand_dims(inputs, 0)
        _, q_hidden = encoder(inputs, hidden)

        r2 = self.calculator.calculate_r2(p_hidden, q_hidden)

        r3 = self.calculator.calculate_r3(utterances[-1], utterances[-3], utterances[-2], self.lang)

        a1, a2, a3 = 0.25, 0.25, 0.5

        return a1 * r1 + a2 * r2 + a3 * r3

    def get_logits(self):
        logits = 0
        return logits


calc = RewardCalculator(encoder, decoder, encoder, decoder, encoder, decoder, ["i don't know", "meh", "huh"])
simulation = Simulation(encoder, decoder, language, max_length, calc)

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
        cumulative = cumulative * gamma + episode_rewards[i]
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

while epoch < num_epochs + 1:

    with tf.GradientTape() as tape:
        # Gather training data
        states_mb, actions_mb, rewards_of_batch, discounted_rewards_mb, nb_episodes_mb = make_batch(batch_size)

        # Backprop
        loss_, _ = sess.run([PGNetwork.loss, PGNetwork.train_opt],
                            feed_dict={PGNetwork.inputs_: states_mb.reshape((len(states_mb), 84, 84, 4)),
                                       PGNetwork.actions: actions_mb,
                                       PGNetwork.discounted_episode_rewards_: discounted_rewards_mb
                                       })

    print("Training Loss: {}".format(loss_))

    variables = encoder.variables + decoder.variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

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
    print("Epoch: ", epoch, "/", num_epochs)
    print("-----------")
    print("Number of training episodes: {}".format(nb_episodes_mb))
    print("Total reward: {}".format(total_reward_of_that_batch, nb_episodes_mb))
    print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
    print("Average Reward of all training: {}".format(average_reward_of_all_training))
    print("Max reward for a batch so far: {}".format(maximumRewardRecorded))

    # Write TF Summaries
    summary = sess.run(write_op, feed_dict={PGNetwork.inputs_: states_mb.reshape((len(states_mb), 84, 84, 4)),
                                            PGNetwork.actions: actions_mb,
                                            PGNetwork.discounted_episode_rewards_: discounted_rewards_mb,
                                            PGNetwork.mean_reward_: mean_reward_of_that_batch
                                            })

    # summary = sess.run(write_op, feed_dict={x: s_.reshape(len(s_),84,84,1), y:a_, d_r: d_r_, r: r_, n: n_})
    writer.add_summary(summary, epoch)
    writer.flush()

    # Save Model
    if epoch % 10 == 0:
        saver.save(sess, "./models/model.ckpt")
        print("Model saved")
    epoch += 1

