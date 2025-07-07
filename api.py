import json
from flask import Flask, jsonify, request
from src.assistant import ChatbotAssistant

app = Flask(__name__)

# assistant
assistant = ChatbotAssistant("intents.json")

# try to load the training data or train if File not found
try:
    assistant.load_model(model_path="chatbot_model.pth", dimension_path="dimensions.json")
    assistant.parse_intents()
except FileNotFoundError:
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(8, 0.001, 100)

    assistant.save_model("chatbot_model.pth", "dimensions.json")

# endpoint for chatting with the model
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

    return jsonify({
        "response" :  assistant.process_message(message)
        })