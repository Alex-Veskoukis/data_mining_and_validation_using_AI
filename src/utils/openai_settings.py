import os
import openai

# OpenAI deployment settings
OPENAI_DEPLOYMENT = "gpt-4.1-mini"
PROMPT_PRICE_PER_1000_TOKENS     = 0.00040
COMPLETION_PRICE_PER_1000_TOKENS = 0.00160

def configure_openai():
    """
    Configure OpenAI API settings.
    """
    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_base or not api_key:
        missing = [
            name for name, value in (
                ("OPENAI_API_BASE", api_base),
                ("OPENAI_API_KEY", api_key),
            ) if not value
        ]
        missing_vars = ", ".join(missing)
        raise RuntimeError(f"Missing OpenAI environment variables: {missing_vars}")

    openai.api_type = "azure"
    openai.api_base = api_base
    openai.api_version = "2025-01-01-preview"
    openai.api_key = api_key
