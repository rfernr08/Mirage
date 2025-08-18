from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# === Cargar modelo y tokenizer ===
model_path = "models\dccuchile_bert-base-spanish-wwm-cased_combinado_final"  # carpeta donde guardaste tu modelo
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predecir_paciente_bert(diagnosticos):
    texto = " ".join(diagnosticos)  # concatenar diagnósticos
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        pred = np.argmax(probs)
    
    print("\n📊 Diagnósticos introducidos:")
    print(diagnosticos)
    print("\n🤖 Predicción del modelo BERT:")
    print(f"Clase predicha: {pred} ({'F20' if pred==1 else 'F20.89'})")
    print(f"Probabilidades: {probs}")

# === Ejemplo ===
nuevo_paciente = [
    "F20.9",  # código
    "Esquizofrenia, no especificada",
    "Uso prolongado de anticoagulantes",
    "Hiperprolactinemia",
    "Trastorno depresivo mayor"
]

predecir_paciente_bert(nuevo_paciente)
