import os
import json
import random

import numpy as np
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import Model 

class ChatbotAssistant:
    def __init__(self, intents_path, function_mapping = None):
        self.model = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self.function_mapping = function_mapping

        self.X = None                   # X is usually a matrix
        self.y = None                   # y is usually a vector

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()

        # taking a sentence, splitting it into individual words -> then reduce these words into stemwords
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words
    
    @staticmethod
    def bag_of_words(words, vocabulary):
        # going through all the word we know and checking if the current word is in the current question
        # [0,0,0,0,0,0,1,0,0,0,1,1,0] - What is microbiome?
        return [1 if word in words else 0 for word in vocabulary]

    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as file:
                intents_data = json.load(file)

            
            for intent in intents_data["intents"]:
                if intent["tag"] not in self.intents:
                    self.intents.append(intent["tag"])
                    self.intents_responses[intent["tag"]] = intent["responses"]

                for pattern in intent["patterns"]:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent["tag"]))

                self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]     # pattern words
            bag = self.bag_of_words(words, self.vocabulary)

            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)
        
        self.X = np.array(bags)         # bags of words
        self.y = np.array(indices)      # correct predictions

    def train_model(self, batch_size, learning_rate, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)       # only one number per instance

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = Model(self.X.shape[1],     # size of an individual bag of words
                            len(self.intents)) 
        # the output of the neural network is the probability of every single intent

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)           # output with random weights and biases
                loss = criterion(outputs, batch_y)      # how wrong is our prediction
                loss.backward()                         # take the loss and propagate it through the network, so we get the gradient and error
                optimizer.step()                        # take a step into the current direction - step largeness depends on the learning rate
                running_loss += loss    
            
            print(f"Epoch {epoch+1}: Loss: {running_loss/len(loader):.4f}")

    def save_model(self, model_path, dimension_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimension_path, "w") as file:
            json.dump({
                'input_size': self.X.shape[1],
                'output_size': len(self.intents)
                       }, file)
            
    def load_model(self, model_path, dimension_path):
        with open(dimension_path, "r") as f:
            dimensions = json.load(f)

        self.model = Model(input_size=dimensions["input_size"], output_size=dimensions["output_size"])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(input_message, self.vocabulary)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_index = torch.argmax(predictions, dim=1).item()          # index with the highest activation
        predicted_intent = self.intents[predicted_index]

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return None
