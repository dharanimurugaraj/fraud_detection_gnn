# 🚨 SecureFlow – Real-Time Fraud Detection using GNNs

[![Streamlit App](https://img.shields.io/badge/Launch-App-green?style=flat&logo=streamlit)](https://secureflow.streamlit.app/)

SecureFlow is an intelligent and lightweight fraud detection app that leverages the power of **Graph Neural Networks (GNNs)** to detect financial fraud in real-time using transaction graph patterns. Designed with performance, clarity, and interpretability in mind, this app is your first step toward smarter and more secure digital transactions.

---

## 💡 **What is SecureFlow?**

SecureFlow analyzes financial transactions as graph structures—where nodes represent users/accounts and edges represent money transfers. Using GNNs, it learns suspicious patterns that traditional ML models might miss.

It is built to be:
- ✅ **Fast** – Real-time predictions, no GPU required
- ✅ **Interpretable** – Visualizes transaction outcomes
- ✅ **Deployable** – Hosted via Streamlit, easy to expand

---

## 🚀 **Live Demo**

🟢 **Try it now:** [secureflow.streamlit.app](https://secureflow.streamlit.app/)

---

## 🧠 **Core Features**

- 🔍 **Fraud Detection using GNNs**  
  Analyzes transaction graphs using PyTorch Geometric models to catch fraudulent behavior.

- 📈 **Probability Visualizer**  
  Clearly shows fraud probability with confidence scores for each transaction.

- 💻 **Streamlit-Based UI**  
  Clean, intuitive interface for fast prototyping and presentations.

- 🧩 **Modular Design**  
  Easily extendable to integrate LLMs, simulators, or API endpoints.

---

## 🧰 **Tech Stack**

- **Frontend**: `Streamlit`  
- **ML Framework**: `PyTorch`, `Torch Geometric`  
- **Data Tools**: `Pandas`, `NumPy`, `Scikit-learn`  
- **Deployment**: Streamlit Cloud  
- **Environment**: CPU-only, lightweight setup

---

## 🛠️ **Installation & Local Setup**

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/SecureFlow.git
cd SecureFlow

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py

