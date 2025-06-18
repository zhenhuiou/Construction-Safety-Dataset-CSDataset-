import os
import pandas as pd
import openai
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# ==== Configurations ====
API_KEY = os.getenv("OPENAI_API_KEY") 
MODEL = "gpt-4"
INPUT_CSV = "prompts_50.csv"
OUTPUT_CSV = "predictions_50.csv"

def query_llm(prompt, model=MODEL):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Error querying LLM:", e)
        return "error"

def extract_label(text):
    for token in text.split():
        if token.strip().isdigit():
            return int(token.strip())
    return -1 

def main():
    openai.api_key = API_KEY

    df = pd.read_csv(INPUT_CSV)
    predictions = []

    print("Querying LLM...")
    for prompt in tqdm(df['input']):
        prediction = query_llm(prompt)
        label = extract_label(prediction)
        predictions.append(label)

    df['prediction'] = predictions
    df.to_csv(OUTPUT_CSV, index=False)

    # Evaluation
    print("\nEvaluation Results:")
    y_true = df['output']
    y_pred = df['prediction']
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()
