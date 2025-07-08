# This project uses a simple neural network model to try to answer questions

# How it works
## 1. Training phase
 - The model is trained based on patterns and responses in the intents.json file
 - Builds a vocabulary from all the patterns from intents.json, unique and stemwords
 - Creates a bag of words: each pattern sentence is converted into a numerical vector
   same length as the vocabulary and value 1 if a word is in the pattern value 0 elsewise
 - A simple neural network is trained to associate these "bag of words" vectors with their
   corresponding tag ("payment", "greeting") and saves it

## 2. Responding
 - Processes user input: same process: tokenized, lemmatized (stemwords) and converts into 
   bag of words
 - This bag of words vector is fed into the trained network and the network predicts which
   tag the input message most likely belongs to
 - Then the system chooses a random respond associated with the predicted tag

# How to run
## 1. Download docker (docker desktop using windows)

## 2. run the following commands
    git clone https://github.com/dsdsgege/Chatbot.git
    cd Chatbot
    docker build -t chatbot-api .
    docker run -d -p 5000:5000 --name chatbot-container chatbot-api
now you can access the chat on http://localhost:5000/