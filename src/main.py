import os
from assistant import ChatbotAssistant


if __name__ == "__main__":
    assistant = ChatbotAssistant("intents.json")
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(8, 0.001, 100)

    assistant.save_model("chatbot_model.pth", "dimensions.json")

    while True:
        message = input("")
        
        if(message == "bye"):
            break

        print(assistant.process_message(message))