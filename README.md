# 💸 Personal Finance Advisor - ML Web App

A smart ML-powered web app that classifies your financial habits as **Overspending**, **Balanced**, or **Saver** — and gives personalized financial advice based on your monthly income and expenses.

Built with Python (Flask), HTML/CSS, Chart.js, and trained using a Random Forest Classifier.

---

## 🧠 Logic Tree (Simplified)

This is the core logic used to label the dataset and guide model decisions:

START
|
|-- Is Income < ₹40,000?
| |-- Yes:
| | |-- Is Rent > 50% of Income?
| | | |-- Yes → OVERSENDING
| | | |-- No → BALANCED
| |
| |-- No:
| |-- Are Total Expenses < 50% of Income?
| |-- Yes → SAVER
| |-- No:
| |-- Is Entertainment + Transport > 20%?
| |-- Yes → BALANCED
| |-- No → SAVER


> ⚠️ Final logic is learned by a RandomForest model, this tree is a simplified idea of how decisions are made.

---

## 🧾 Input Features Used

- Income
- Age
- Dependents
- Occupation
- City_Tier
- Rent
- Groceries
- Transport
- Entertainment
- Utilities
- Healthcare

---

## 🧠 Model Info

- Model: `RandomForestClassifier` (150 estimators)
- Dataset: Custom balanced dataset (3 labels: Overspending, Balanced, Saver)
- Accuracy: ~88–92%
- Preprocessing: OneHotEncoding for categorical features (Occupation, City_Tier)

---

## 🖥️ Web App Features

- 🔍 Predict your financial category
- 📊 Chart showing monthly spending breakdown
- 💡 Personalized financial tips
- ✨ Neon-glass UI with sci-fi feel


 
