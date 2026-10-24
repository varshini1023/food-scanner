# 🍽️ Food Scanner

A machine learning-based web application that identifies South Indian food from images and provides nutritional information to help users make informed food choices.

## ✨ Features

* **Food Recognition:** Upload a food image to identify the dish using a trained TensorFlow model.
* **Nutrition Information:** View estimated nutritional values for recognized foods.
* **BMI Calculator:** Calculate Body Mass Index using height and weight.
* **User Authentication:** Register and log in to access application features.
* **Healthy Food Suggestions:** Explore food recommendations and protein-rich options.
* **Web Interface:** Built with Flask, HTML, CSS, and JavaScript.

## 🛠️ Technologies Used

* Python
* Flask
* TensorFlow and Keras
* NumPy
* Pillow
* Flask-SQLAlchemy
* SQLite
* HTML and CSS

## 📁 Project Structure

```text
food-scanner/
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── food_classifier_v4.keras
├── food_class_names.txt
├── food_class_names_v5.txt
├── static/
├── templates/
└── README.md
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/varshini1023/food-scanner.git
cd food-scanner
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## ⚠️ Disclaimer

Nutritional values are estimates and may vary depending on ingredients, portion sizes, and preparation methods. Food recognition predictions may not always be accurate.

## 👩‍💻 Project Status

Developed as a student project to explore machine learning, image classification, and web application development.
