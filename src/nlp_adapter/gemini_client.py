# src/nlp_adapter/gemini_client.py
"""
GeminiAdapter
- Lightweight wrapper over google.genai client for simple chat usage
- Accepts optional api_key (falls back to env-based auth if None)
"""
from google import genai


class GeminiAdapter:
    def __init__(self, model: str = "gemini-2.5-flash-lite", api_key: str | None = None):
        """
        Initialize Gemini client + chat.
        If api_key is provided, genai.Client(api_key=api_key) will be used.
        """
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()
        self.model = model
        self.chat = self.client.chats.create(model=model)

    def send_message(self, message: str) -> str:
        """
        Send a message to the current Gemini chat and return response text.
        """
        resp = self.chat.send_message(message)
        return resp.text

    def get_history(self):
        """
        Return conversation history as list of {role, text}
        """
        out = []
        for m in self.chat.get_history():
            out.append({"role": m.role, "text": m.parts[0].text})
        return out

    def reset_session(self, model: str | None = None):
        """
        Reset the chat session (new conversation).
        """
        if model is None:
            model = self.model
        self.chat = self.client.chats.create(model=model)
