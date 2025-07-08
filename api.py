import json
import os
from flask import Flask, render_template, jsonify, request
from src.assistant import ChatbotAssistant
from flask_cors import CORS, cross_origin

app = Flask(__name__)

# assistant
assistant = ChatbotAssistant("intents.json")

"""
load the training data from saved model or train if File not found
"""
try:
    assistant.load_model(model_path="chatbot_model.pth", dimension_path="dimensions.json")
    assistant.parse_intents()
except FileNotFoundError:
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(8, 0.001, 100)

    assistant.save_model("chatbot_model.pth", "dimensions.json")

""" 
endpoint for chatting with the model
"""
@cross_origin() # enables request from other port
@app.route("/chat", methods = ['POST'])
def chat():
    if assistant is None:
        return jsonify({
            "error" : "A szolgáltatás nem működik"
            })
    
    # get data from request object
    data = request.get_json()
    if not data or "message" not in data.keys():
        return jsonify({
            "error" : "Nem megfelelő kérés"
            })
    
    message = data["message"]
    print(message)
    try:
        response = jsonify({
            "response" :  assistant.process_message(message)
            })
    
    except RuntimeError:
        os.remove("chatbot_model.pth")
        assistant.parse_intents()
        assistant.prepare_data()
        assistant.train_model(8, 0.001, 100)

        assistant.save_model("chatbot_model.pth", "dimensions.json")
        response = jsonify({
            "response" :  assistant.process_message(message)
            })
    
    return response

"""
chat window
"""
@app.route("/", methods= ["GET"])
def chatWindow():
    return render_template("chat.html")