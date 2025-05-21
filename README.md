# 🛢️ Project Spartan (Spread Analyzer Only) – Installation Guide

This guide will walk you through setting up and running **Project Spartan** step-by-step.

---

## 📥 Step 1: Install Git

1. Download Git from:
   👉 [https://git-scm.com/downloads](https://git-scm.com/downloads)
2. Follow the installation instructions for your operating system.

---

## 📥 Step 2: Install Anaconda

1. Download Anaconda from:
   👉 [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
2. Follow the installation instructions for your operating system.

---

## 📁 Step 3: Navigate to Your Documents Folder

Before cloning the repository, navigate to your **Documents** folder where you want to store the project.

1. Open **Anaconda Prompt** (on Windows) or your **terminal** (on macOS/Linux).
2. Run the following command to go to your Documents folder:

   **For Windows:**

   ```bash
   cd %USERPROFILE%\Documents
   ```

   **For macOS/Linux:**

   ```bash
   cd ~/Documents
   ```

This ensures that the project is stored in your **Documents** folder.

---

## 📂 Step 4: Clone the Project Repository

Now that you're in your **Documents** folder:

1. Run the following command to clone the repository:

   ```bash
   git clone https://github.com/mukeshcodeshere/SpreadRepoOnly.git
   ```

2. Navigate into the project folder:

   ```bash
   cd SpreadRepoOnly
   ```

   The project will now be stored in your **Documents** folder, inside the `SpreadRepoOnly` directory.

---

## 🐍 Step 5: Create & Activate Your Python Environment

In Anaconda Prompt (or terminal), run these commands to create and activate a new environment:

```bash
conda create --name work python=3.13.2
conda activate work
```

This sets up a clean Python environment named `work`.

---

## 📦 Step 6: Install Project Dependencies

Make sure you're inside the `SpreadRepoOnly` folder, then install the required dependencies by running:

```bash
pip install -r requirements.txt
```

---
Here's your updated section incorporating the extended and refined information:

---

## ▶️ Step 7: Run the Application

To start the application, run the following command in your terminal:

```bash
streamlit run raul_seasonality.py

OR

python -m streamlit run raul_seasonality.py
```