# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 20:33:23 2019

@author: dalai
"""

# Importing the libraries
import numpy as np
import tensorflow as tf
import re # For cleaning and replacing the texts of the corpus
import time # To measure the training time
 
 
 
# Data Preprocessing Stage 
 
 
 
# Step 1: Importing the dataset
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
 
# Step 2: Dictionary mapping each line to its respective ID
id2line = {} # Initializing an empty dictionary
for line in lines:
    _line = line.split(' +++$+++ ') # Splitting or segregating the dataset wrt +++$+++
    if len(_line) == 5:  # NOT NECESSARY, but it is just to check if the no of parameters in the dataset are same (here 5) so as to prevent shifting effect
        id2line[_line[0]] = _line[4] # Since we only need the first and the last column
 
# Step 3: Creating a list of all of the conversations
conversations_ids = [] # Initializing an empty list
for conversation in conversations[:-1]: # Since the last row of the dataset is an empty row so iterating through complete dataset except last row
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "") # Splitting the dataset wrt to ' +++$+++ ', just taking the last column, and removing all the punctuations and brackets from it except comma
    conversations_ids.append(_conversation.split(',')) # The reason why comma wasn't replaced in the previous step was to split the dataset wrt it
 
# Step 4: Separating the questions and the answers into different lists
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1): # Since upper-bound in python is not included
        questions.append(id2line[conversation[i]]) # The first identifier index in the dataset is the question and
        answers.append(id2line[conversation[i+1]]) # the following identifiers are the answers
 
# Step 5: Cleaning the texts using RE library
def clean_text(text): # Making a function so as to clean the dataset
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text

# Step 6: Cleaning the lists of questions and answers using the function made in previous step 
# Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question)) 
# Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
 
# Step 7: Filtering out the questions and answers depending on there size so as to efficiently train the chatbot
short_questions = []
short_answers = []
i = 0
for question in clean_questions:
    if 2 <= len(question.split()) <= 25: # Taking a range from 2 to 25 for the training of dataset (since both short will be deficient and long data will be overwhelming for chatbot to learn)
        short_questions.append(question)
        short_answers.append(clean_answers[i])
    i += 1 # Incrementing the value of i so that it iterates each and every conversation
# In the above step, we filter from the questions and include the subsequent answers
clean_questions = []
clean_answers = []
i = 0
for answer in short_answers:
    if 2 <= len(answer.split()) <= 25:
        clean_answers.append(answer)
        clean_questions.append(short_questions[i])
    i += 1
# In the above step, we filter from the answers and include the subsequent questions
 
# Step 8: Dictionary mapping each words to its no. of occurence in the dataset
word2count = {} 
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1 # If the word is not present in dictionary, initiate its count
        else:
            word2count[word] += 1 # If the word is already included in the dictionary, increment the count by 1
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
 
# Step 9: Dictionary mapping the words in the questions and the answers to a unique integer
threshold_questions = 15 # Creating a threshold so as to include only the words with occurences greater than or equal to 15
questionswords2int = {}
word_number = 0 
for word, count in word2count.items():
    if count >= threshold_questions:
        questionswords2int[word] = word_number
        word_number += 1
threshold_answers = 15 # Instead of using two different variables for threshold, we can use just a single variable
answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_answers:
        answerswords2int[word] = word_number
        word_number += 1
 
# Step 10: Adding the tokens to questionswords2int and answerswords2int dictionary
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1 # Since upper-bound value is excluded, so +1 to include the upper_bound value
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1
 
# Step 10: Inverting the dictionary of answerswords2int dictionary
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}
 
# Step 11: Adding <EOS> to end of every answer in the dictionary
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'
 
# Step 12: Translating all the questions and the answers into integers and replacing all the filtered out words with <OUT> 
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split(): # Traversing each and every word present in the clean_questions list
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)
 
# Step 13: Sorting questions and answers in ascending order so as to ease the training of chatbot
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1): # Even though the range is from 1 to 25, but the questions with minimum words will have atleast 2 words due to previous constraints
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
 
 
 
# Building the Seq2Seq Model
 
 
 
# Step 1: Creating placeholders for the inputs and the targets 
def model_inputs(): # Function for defining tensorflow placeholders
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input') 
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob
 
# Step 2: Preprocessing the targets (because the decoder needs the data to be in a special format for it to decode)
def preprocess_targets(targets, word2int, batch_size): # The targets must be in batches & each of the answers of the target must start with SOS token
    left_side = tf.fill([batch_size, 1], word2int['<SOS>']) # Making a vector of <SOS> token, size being batch_size*1
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1]) # Making a vector targets without the last column and slided element by element
    preprocessed_targets = tf.concat([left_side, right_side], 1) # Concatenating both the vectors horizontally
    return preprocessed_targets
 
# Step 3: Creating the Encoder RNN 
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size) 
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob) # Applying dropout to LSTM so as to drop specific no of neurons during training of chatbot so as train it efficiently
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, # The underscore is initially used because bidirectional rnn gives 2 output but we are interested in just the second one
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state
 
# Step 4: Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size]) # A three dimensional matrix of zeros
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, # Again underscores used due to the similar reasons as mentioned in the above function
                                                                  training_decoder_function,
                                                                  decoder_embedded_input,
                                                                  sequence_length,
                                                                  scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
 
# Step 5: Decoding the validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size]) # A three dimensional matrix of zeros
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                    test_decoder_function,
                                                                    scope = decoding_scope)
    return test_predictions
 
# Step 5: Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1) # Getting weights to be a truncated normal distribution
        biases = tf.zeros_initializer() # Initializing biases value to be 0
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables() # So as to use the same variables as used in the above function
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions
 
# Step 6: Building the final Seq2Seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
 
 
 
# Training the Seq2Seq model created previously
    
 
# Step 1: Setting the Hyperparameters
epochs = 100
batch_size = 32
rnn_size = 1024
num_layers = 3
encoding_embedding_size = 1024
decoding_embedding_size = 1024
learning_rate = 0.001
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5
 
# Step 2: Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()
 
# Step 3: Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()
 
# Step 4: Setting the sequence length (will be max of the data length in dataset)
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')
 
# Step 5: Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)
 
# Step 6: Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)
 
# Step 7: Setting up the Loss Error and the Optimizer with Gradient Clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None] # Clipping the gradient from -5 to 5
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
 
# Step 8: Padding the sequences with appropiate no of <PAD> token
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences]) # Max length data used so as to get a proper padding of data so that each data is of same and equal length
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]
 
# Step 9: Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch
 
# Step 10: Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]
 
# Step 11: Training the Chatbot
batch_index_check_training_loss = 100 # Checking training loss every 100 epochs
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1 # Checking validation loss every half of batch size
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0 # This value gets incremented by 1 if the loss error doesn't increase with each iteration
early_stopping_stop = 100 # Used to check if the early_stopping_check becomes its equal, if yes, then we break the process
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I am getting better as time goes by.')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("I still need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("I can't speak any better, this is the best I can do.")
        break
print("Over")
 
 
 
# Testing the Seq2Seq Model
 
 
 
# Step 1: Loading the weights and Running the session
checkpoint = "./chatbot_weights.ckpt" # Loading the weights after training from .ckpt file
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)
 
# Step 2: Converting the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]
 
# Step 3: Setting up the chat
while(True):
    question = input("You: ")
    if question == 'Goodbye':
        break
    question = convert_string2int(question, questionswords2int)
    question = question + [questionswords2int['<PAD>']] * (25 - len(question))
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answersints2word[i] == 'i':
            token = ' I'
        elif answersints2word[i] == '<EOS>':
            token = '.'
        elif answersints2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answersints2word[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)