# chat_bot
## Chat bot using Tensorflow Keras

## Sample answers predicted by the chat bot

### >>> import robo.py as r
### >>> r.askme()

Enter question : hi

hello

Enter question : what is your age

i am still young by your standards

Enter question : what is your shoe size

have you ever heard of software with shoes

Enter question : what is ai

artificial intelligence is the branch of engineering and science devoted to constructing machines that think

Enter question : what you can do for me

i have read just in binary

Enter question : can you walk

the plan for my body includes legs but they are not yet built

## Steps involved:

1. Read the data ans seperate questions and answers.

2. Tokenize and pad the questions. ( encoder input data ), also used glove word embedding.

3. Tokenize and pad the answers. Append <START> and <END> in all sequences.( decoder input data )

4. Tokenize and pad the answers. Remove the <START> in all sequences. One hot encode the sequences. ( decoder target data )
  
5. saved encoder_input_data, decoder_input_data, decoder_target_data, embedding_matrix and tokenizer for future use.

6. Prepare the model and save it.

7. Make interface model and save encoded and decoded model for future use.

8. Preapre function for question and answer.
