"""
LLM Service - Handles all communication with language models.

Single Responsibility: This file ONLY deals with LLM communication.
If we switch from Groq to OpenAI, this is the ONLY file that changes.
"""

import re

from groq import Groq

from app.config import settings


class LLMService:

    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.llm_model

    def _strip_think_tags(self, text: str) -> str:
        """Remove Qwen's <think>...</think> reasoning from output."""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = cleaned.strip()

        # If stripping removed everything, the model's actual answer
        # was inside the think block. Return raw response as fallback.
        if not cleaned:
            # Remove just the tags, keep the content
            fallback = text.replace("<think>", "").replace("</think>", "")
            return fallback.strip()

        return cleaned

    def chat(
        self,
        user_message: str,
        system_prompt: str = None,
        conversation_history: list = None,
    ) -> str:
        """
        Send a message to the LLM and get a response.

        Args:
            user_message: The current user message
            system_prompt: LLM behavior instructions
            conversation_history: Previous messages in the conversation.
                Each item is a dict: {"role": "user"|"assistant", "content": "..."}

        Returns:
            The LLM's response as a string
        """

        if system_prompt is None:
            system_prompt = (
                "You are a helpful financial research assistant. "
                "Provide accurate, well-structured answers. "
                "If you are unsure about something, say so clearly."
            )

        # Build messages: system prompt first, then history, then current message
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if it exists
        if conversation_history:
            messages.extend(conversation_history)

        # Add the current user message
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
        )

        raw_response = response.choices[0].message.content
        return self._strip_think_tags(raw_response)