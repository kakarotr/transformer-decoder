import os


def get_model(provider: str):
    base_url = os.environ[f"{provider.upper()}_BASE_URL"]
    api_key = os.environ[f"{provider.upper()}_API_KEY"]
    model = os.environ[f"{provider.upper()}_MODEL"]
    return base_url, api_key, model
