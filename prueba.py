import requests

url = "http://localhost:8000/ask"
payload = {
    "questions": [
        "What is crop rotation and why is it important?",
        "How can soil pH affect plant growth?",
        "What are effective water management practices in agriculture?"
    ]
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    print("Respuesta de la API:")
    print(response.json())
else:
    print("Error:", response.status_code, response.text)
