#!/usr/bin/env python
# coding: utf-8

# In[75]:



    
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras import layers , activations , models , preprocessing

encoder_input_data = np.load('encoder_input_data.npy')
decoder_input_data = np.load('decoder_input_data.npy')
decoder_target_data = np.load('decoder_target_data.npy')

embedding_matrix = np.load('embedding_matrix.npy' ) 

tokenizer = pickle.load( open('tokenizer' , 'rb'))

num_tokens = len( tokenizer.word_index )+1
word_dict = tokenizer.word_index

max_question_len = encoder_input_data.shape[1]
max_answer_len = decoder_input_data.shape[1]

#model = tf.keras.models.load_model('model_chatbots1.h5')

enc_model = tf.keras.models.load_model( 'enc_model.h5' )


dec_model = tf.keras.models.load_model( 'dec_model.h5' )
    
    
def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append( word_dict[ word ] ) 
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=max_question_len , padding='post')
    
def askme():
    for _ in range(6):
        try:
            states_values = enc_model.predict( str_to_tokens( input( 'Enter question : ' ) ) ) #(1,200)
            # array([[ 0.43804294, -0.37364703,  0.15553352,  0.30735624,  0.9615902 ,...]]
            empty_target_seq = np.zeros( ( 1 , 1 ) )
            empty_target_seq[0, 0] = word_dict['start']    # array([[1.]])
            stop_condition = False
            decoded_translation = ''
            while not stop_condition :
                dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
                sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
                sampled_word = None
                for word , index in word_dict.items() :
                    if sampled_word_index == index :
                        decoded_translation += ' {}'.format( word )
                        sampled_word = word
        
                if sampled_word == 'end' or len(decoded_translation.split()) > max_answer_len:
                    stop_condition = True
            
                empty_target_seq = np.zeros( ( 1 , 1 ) )  
                empty_target_seq[ 0 , 0 ] = sampled_word_index
                states_values = [ h , c ] 

            ans = [w for w in decoded_translation.split() if not w in ['end']]
            ans = ' '.join(ans)
            print(ans)
        except Exception as e:
            print("sorry, can't answer")


