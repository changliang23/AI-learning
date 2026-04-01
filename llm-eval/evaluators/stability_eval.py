from models.openai_model import call_model

def stability(prompt, k=3):
    results = [call_model(prompt) for _ in range(k)]
    unique = len(set(results))
    return unique / k, results