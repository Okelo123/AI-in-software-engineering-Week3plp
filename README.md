# AI-in-software-engineering-Week3plp

📊 Iris & MNIST Classification using Classical ML and Deep Learning
🧠 Week 3 Assignment – AI in Software Engineering
This project demonstrates supervised learning techniques using both classical machine learning and deep learning frameworks. It aims to explore the strengths, use-cases, and accuracy of Scikit-learn, TensorFlow, and PyTorch across two tasks:

Task 1: Iris Species Classification

Task 2: MNIST Digit Recognition

📁 Files Included
Week3Assignment14.ipynb: Jupyter notebook containing code for all tasks

README.md: Project overview and instructions

(Optional) requirements.txt: Python dependencies

🔍 Project Overview
✅ Task 1: Iris Species Classification
Framework	Type	Model	Accuracy
Scikit-learn	Classical ML	Decision Tree	✅ 100%
TensorFlow	Deep Learning	Dense Neural Net	⚠️ ~56.7%
PyTorch	Deep Learning	Dense Neural Net	✅ 100%

Key Steps:
Data cleaning & preprocessing

Label encoding & train-test split

Model training (tree-based and NN)

Evaluation using: Accuracy, Precision, and Recall

✅ Task 2: MNIST Digit Classification
Framework: TensorFlow CNN

Dataset: 70,000 grayscale images of handwritten digits (28x28)

Accuracy Achieved: ✅ ~98.6%

CNN Architecture:
2 × Conv2D + MaxPooling layers

1 × Dense (ReLU) + Output (Softmax)

📌 Comparative Summary
Feature	Scikit-learn	TensorFlow	PyTorch
Type	Classical ML	Deep Learning	Deep Learning
Learning Curve	Beginner-friendly	Intermediate	Intermediate/Advanced
Deployment	Joblib, ONNX	TF Lite, TF.js	TorchScript, ONNX
Best Use Case	Tabular data	Images, NLP	Flexibility in research

🎯 Scikit-learn excels in tabular ML tasks like Iris.
🤖 TensorFlow/PyTorch are better suited for complex data like images (e.g., MNIST).

⚠️ Debugging Neural Networks: A Learning Exercise
A buggy TensorFlow classification model was reviewed. Issues included:

❌ No softmax activation

❌ Wrong loss function (MSE for classification)

❌ Undefined training data

A corrected version applied:

Proper data loading

categorical_crossentropy loss

Final softmax activation

One-hot encoding

⚖️ Ethical AI Analysis
Concern	Solution
🔍 Bias	Use diverse datasets, fairness audits
🔐 Privacy	Anonymize data, secure storage
🧠 Explainability	Use tools like SHAP, LIME
🌱 Sustainability	Optimize model size and training

✅ Responsible Practices:
Use interpretable models where possible

Apply human-in-the-loop reviews

Follow ethical AI guidelines (e.g., Google, IBM)

🚀 How to Run This Project
Open in Google Colab or run locally in Jupyter

Ensure required libraries are installed:

bash
Copy
Edit
pip install scikit-learn tensorflow torch torchvision
Execute cells sequentially.

🧾 Dependencies
You can generate a requirements.txt file with:

bash
Copy
Edit
pip freeze > requirements.txt
🤝 Credits
Dataset: UCI Iris Dataset & MNIST (Keras)

Tools: Python, Jupyter, Scikit-learn, TensorFlow, PyTorch


