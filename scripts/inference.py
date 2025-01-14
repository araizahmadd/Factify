from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('/content/results/checkpoint-3852')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Function to make predictions
def predict_statement(statement):
    # Tokenize the input statement
    inputs = tokenizer(statement, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    # Get the model's prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = logits.argmax(-1).item()  # Get the predicted label (index of the highest logit)
    
    # Mapping of labels to categories (as per LIAR dataset)
    labels = ['False', 'Half-True', 'Mostly-True', 'True', 'Barely-True', 'Pants-Fire']
    
    # Return the predicted label
    return labels[prediction]

# Example usage
statement = input("please add a fact")
prediction = predict_statement(statement)
print(f"Prediction: {prediction}")

