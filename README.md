# ✈️ OC - Projet 7 : Analyse de sentiments sur les réseaux sociaux

Ce projet est réalisé dans le cadre du parcours *AI Engineer* d’OpenClassrooms, en collaboration avec **Air Paradis** (cas d’usage fictif).  
L’objectif est de développer un prototype d’IA capable de prédire le **sentiment d’un tweet**, en s’appuyant sur une **démarche MLOps rigoureuse**.

---

## 🎯 Objectifs

1. Prototyper un modèle de **prédiction de sentiment** applicable aux réseaux sociaux.
2. Déployer ce modèle via une **API FastAPI** accessible en ligne.
3. Intégrer une **interface utilisateur (Streamlit)** permettant de tester la prédiction.
4. Mettre en œuvre une **démarche MLOps complète** :
   - gestion des expériences via MLFlow,
   - CI/CD pour déploiement continu,
   - suivi de performance en production via **Azure Application Insights**,
   - déclenchement d’alertes sur dérive du modèle.

---

## 📦 Approches testées

### 1. Modèle sur mesure simple
- TF-IDF + modèle de régression logistique, RandomForest, XGBoost.
- Avantage : rapide, interprétable, léger.

### 2. Modèle sur mesure avancé
- Embeddings pré-entraînés (GloVe, FastText) + RNN (GRU/LSTM bidirectionnels).
- Implémentation sous **TensorFlow/Keras**.

### 3. Modèle BERT
- Utilisation de **HuggingFace Transformers** pour classification binaire.
- Analyse comparative des performances vs. autres modèles.

---

## 🛠️ Architecture technique

```text
[ Tweet utilisateur ]
        │
        ▼
[ Interface Streamlit ] ←→ [ API FastAPI hébergée sur Azure ]
        │                                │
        ▼                                ▼
Validation utilisateur         [ Modèle GRU + Embeddings ]
(feedback/alerte)              [ MLFlow + Logs Azure ]
```

---

## 🔁 Suivi en production (MLOps)

- Utilisation de **MLFlow** pour le suivi des expériences.
- Sérialisation du modèle (TensorFlow) et du tokenizer.
- Déploiement automatique sur **Azure Webapp (plan gratuit)**.
- Interface de test utilisateur via **Streamlit**.
- Journalisation des prédictions invalidées dans **Azure Application Insights**.
- Système d’alerte configurable en cas de dérive.

---

## 📁 Arborescence du dépôt

```
├── 2_scripts_notebook_modélisation_012025.ipynb → Modélisation + comparaisons
├── 4_interface_test_API_012025.py               → Interface Streamlit connectée à l’API
├── 1_API_012025.url                              → Lien vers l'API déployée
├── 3_dossier_code_012025.url                     → Lien vers le code complet sur GitHub
├── 5_blog_012025.pdf                             → Article de blog décrivant les approches et la démarche MLOps
└── README.md                                     → Présentation du projet
```

---

## 🔗 Outils et librairies utilisés

- **Python**, **TensorFlow/Keras**, **scikit-learn**, **transformers**, **MLFlow**
- **FastAPI**, **Streamlit**, **PyTest**, **GitHub Actions**
- **Azure Application Insights** pour la télémétrie

---

## 🧠 Auteur

Projet réalisé par **AnthonyJVID** dans le cadre du parcours *AI Engineer* chez OpenClassrooms.

---

## 📄 Licence

Projet pédagogique basé sur des données Twitter open source. Aucune donnée personnelle réelle n’a été utilisée.
