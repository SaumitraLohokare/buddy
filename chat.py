import requests

def generate_response(model, inputs, max_new_tokens):
    url = "https://chat.petals.dev/api/v1/generate"

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {
        "model": model,
        "inputs": inputs,
        "max_new_tokens": max_new_tokens,
    }

    response = requests.post(url, headers=headers, data=data)

    return response.json()

# Example usage:
model = "stabilityai/StableBeluga2"
inputs = "### User:\nHello, how are you? \n### Assistant:\n"
max_new_tokens = "1000"
result = generate_response(model, inputs, max_new_tokens)
print(result)
