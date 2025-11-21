import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from neo4j import GraphDatabase
import numpy as np
from pyvis.network import Network
import tempfile
import os
import dotenv

# -------------------------
# CONFIGURACIÓN INICIAL
# -------------------------
st.set_page_config(page_title="Asistente Clínico - Esquizofrenia", layout="wide")

# -------------------------
# Conexión Neo4j
# -------------------------
dotenv.load_dotenv()

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

# === Crea el driver y la sesión ===
driver = GraphDatabase.driver(URI, auth=AUTH)


# -------------------------
# Cargar modelo BERT
# -------------------------
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# -------------------------
# Funciones Neo4j + Embeddings
# -------------------------

def get_diagnostic_list():
    q = "MATCH (d:Diagnostico) RETURN d.terminoEN AS name"
    with driver.session() as session:
        result = session.run(q)
        return [record["name"] for record in result]

def get_patient_list():
    q = "MATCH (p:Paciente) RETURN p.numero_historia AS id"
    with driver.session() as session:
        result = session.run(q)
        return [record["id"] for record in result]

def get_patient_diagnostics(pid):
    q = """
    MATCH (p:Paciente {numero_historia:$pid})-[:`DIAGNOSTICO_ASOCIADO`]->(d:Diagnostico)
    RETURN collect(d {.*, embedding:d.embedding}) AS diagnos
    """
    with driver.session() as session:
        r = session.run(q, pid=pid).single()
        return r["diagnos"] if r else []


def get_final_diagnosis_embedding(pid):
    q = """
    MATCH (p:Paciente {numero_historia:$pid})-[:`DIAGNOSTICO_PSIQUIATRICO`]->(d:Diagnostico)
    RETURN d.embedding AS emb
    """
    with driver.session() as session:
        r = session.run(q, pid=pid).single()
        return np.array(r["emb"], dtype=float) if r else None

# -------------------------
# Visualización del Grafo (PyVis)
# -------------------------

def visualize_patient_graph(pid):
    q = """
    MATCH (p:Paciente {numero_historia:$pid})-[r]->(d:Diagnostico)
    RETURN p, r, d
    """
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.barnes_hut()

    with driver.session() as session:
        results = session.run(q, pid=pid)
        for record in results:
            p = record["p"]
            d = record["d"]
            r = record["r"]

            net.add_node(p.id, label=f"Paciente {p['numero_historia']}", color="#00ff00")
            net.add_node(d.id, label=d.get("terminoEN", "Diagnóstico"), color="#00aaff")
            net.add_edge(p.id, d.id, label=r.type)

    tmp_dir = tempfile.gettempdir()
    html_path = os.path.join(tmp_dir, f"graph_{pid}.html")
    net.save_graph(html_path)

    return html_path

# -------------------------
# Funciones BERT
# -------------------------

def bert_predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).numpy()[0]
    pred_class = int(np.argmax(probs))
    return pred_class, probs

# -------------------------
# Diagnóstico simulado por embeddings
# -------------------------

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def simulated_embedding_diagnosis(diags):
    if not diags:
        return "No hay embeddings disponibles para este paciente."

    if any(d.get("embedding") is None for d in diags):
        return "Faltan embeddings en algunos diagnósticos."

    embeddings = [np.array(d["embedding"], dtype=float) for d in diags]
    avg_emb = np.mean(embeddings, axis=0)

    q = "MATCH (d:Diagnostico {tipo:'final'}) RETURN d.nombre_es AS name, d.embedding AS emb"
    with driver.session() as session:
        result = session.run(q)
        sims = []
        for r in result:
            emb = np.array(r["emb"], dtype=float)
            sim = cosine_similarity(avg_emb, emb)
            sims.append((r["name"], sim))

    if not sims:
        return "No existen diagnósticos finales para comparar."

    sims.sort(key=lambda x: x[1], reverse=True)
    best, score = sims[0]
    return f"Diagnóstico simulado sugerido: **{best}** (similitud: {score:.3f})"

# -------------------------
# INTERFAZ PRINCIPAL
# -------------------------
st.title("🧠 Asistente Clínico Experimental para Esquizofrenia")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Selección de Paciente")
    patient_ids = get_patient_list()
    selected_pid = st.selectbox("Paciente:", patient_ids)

    if selected_pid:
        diags = get_patient_diagnostics(selected_pid)
        st.markdown("### Diagnósticos del Paciente")
        if diags:
            for d in diags:
                st.markdown(f"- **{d.get('terminoEN','')}** — `{d.get('ICD10','?')}`")
        else:
            st.info("No se encontraron diagnósticos.")

        st.markdown("---")
        st.subheader("🕸️ Visualización del Grafo del Paciente")
        graph_path = visualize_patient_graph(selected_pid)
        st.components.v1.html(open(graph_path, "r", encoding="utf-8").read(), height=600)

        st.markdown("---")
        st.subheader("🔮 Diagnóstico basado en Embeddings (Simulado)")
        #emb_result = simulated_embedding_diagnosis(diags)
        #st.write(emb_result)

with col2:
    st.subheader("💬 Módulo BERT — Clasificación de Texto Clínico")
    user_input = st.text_area("Escribe una nota o síntoma para clasificar:")

    if st.button("Clasificar con BERT"):
        pred, probs = bert_predict(user_input)
        st.write(f"**Clase predicha:** {pred}")
        st.json({"probs": probs.tolist()})

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("⚙ Estado del Sistema")
st.sidebar.success("Embeddings Neo4j + BERT + Visualización de grafo funcional.")
