import os
import httpx

from config import LLM_API_URL, LLM_API_KEY, LLM_MODEL, LLM_TIMEOUT_SECONDS

api_url = LLM_API_URL
api_key = LLM_API_KEY
model = LLM_MODEL
http_client = httpx.Client(timeout=LLM_TIMEOUT_SECONDS)

def build_prompt(question: str, context: list[str]) -> str:
	joined_context = "\n".join(f"- {item}" for item in context)
	return (
		"You are a domain assistant. Use the provided context to answer the question. "
		"If the context is insufficient, say so clearly.\n\n"
		f"Context:\n{joined_context}\n\n"
		f"Question: {question}"
	)


def generate_answer(question: str, context: list[str]) -> str:
	if not context:
		return "No relevant medical information found."

	if not api_url or not api_key:
		return context[0]

	payload = {
		"model": model,
		"messages": [
			{
				"role": "system",
				"content": "You are a helpful assistant that answers based on provided context.",
			},
			{
				"role": "user",
				"content": build_prompt(question, context),
			},
		],
		"temperature": 0.2,
	}

	headers = {
		"Authorization": f"Bearer {api_key}",
		"Content-Type": "application/json",
	}

	try:
		response = http_client.post(api_url, json=payload, headers=headers)
		response.raise_for_status()
		response_data = response.json()
		content = response_data["choices"][0]["message"]["content"].strip()
		return content or context[0]
	except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
		return context[0]
