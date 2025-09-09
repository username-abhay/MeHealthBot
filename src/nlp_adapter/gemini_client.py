
from google import genai

class GeminiAdapter:
    def __init__(self, model="gemini-2.5-flash-lite"):
        self.client = genai.Client()
        self.chat = self.client.chats.create(model=model)
    
    def send_message(self, message: str) -> str:
        """
        Send a message to Gemini and return the response text.
        """
        response = self.chat.send_message(message)
        return response.text
    
    def get_history(self):
        """
        Returns the full conversation history as a list of messages.
        Each message contains 'role' and 'text'
        """
        messages = []
        for message in self.chat.get_history():
            messages.append({
                "role": message.role,
                "text": message.parts[0].text
            })
        return messages
    
    def reset_session(self, model="gemini-2.5-flash-lite"):
        """
        Reset the Gemini chat session (new session)
        """
        self.chat = self.client.chats.create(model=model)
