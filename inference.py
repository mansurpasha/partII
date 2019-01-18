encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(units,), name="H_state_input")
decoder_state_input_c = Input(shape=(units,), name="C_state_input")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(word_embedding_answer, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
encoder_model.summary()
decoder_model.summary()

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Populate the first character of target sequence with the start character.
    target_seq = np.array([language.word2idx["<start>"]])

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = language.idx2word[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '<end>' or
                    len(decoded_sentence) > max_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.array([sampled_token_index])

        # Update states
        states_value = [h, c]

    return decoded_sentence

print(input_tensor.shape)
print(input_tensor[0].shape)
decode_sequence(input_tensor[0:1])

for i in range(input_tensor.shape[0]):
    print(decode_sequence(input_tensor[i]))
