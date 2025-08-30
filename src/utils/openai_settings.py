import openai

# OpenAI deployment settings
OPENAI_DEPLOYMENT = "<DEPLOYMENT_NAME>"
PROMPT_PRICE_PER_1000_TOKENS     = 0.00015
COMPLETION_PRICE_PER_1000_TOKENS = 0.00060

def configure_openai():
    """
    Configure OpenAI API settings.
    """
    openai.api_type = "azure"
    openai.api_base = "<ENDPOINT>"
    openai.api_version = "<VERSION>"
    openai.api_key = "<KEY>"
