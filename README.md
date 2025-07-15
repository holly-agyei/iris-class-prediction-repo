# Iris Class Prediction App

A modern web application for predicting the species of an Iris flower using machine learning and a beautiful, mobile-friendly UI.

## 🌸 Overview
This app allows users to enter measurements of an Iris flower (sepal length, sepal width, petal length, petal width) and predicts the species (Setosa, Versicolor, or Virginica). The results page provides not only the prediction, but also a photo, fun facts, health benefits, and societal/cultural notes about the predicted flower.

## ✨ Features
- **Beautiful, responsive UI** with glassmorphism and dark mode
- **Mobile-friendly** and easy to use
- **Displays flower image and info cards** for each species
- **Fun facts, health, and cultural notes** for user engagement
- **Local static images** for fast, reliable display

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **Frontend:** HTML, CSS (inline, glassmorphism, flexbox)
- **ML Model:** scikit-learn (Logistic Regression)

## 🚀 How to Run
1. **Clone the repo:**
   ```bash
   git clone https://github.com/holly-agyei/iris-class-prediction-repo.git
   cd iris-class-prediction-repo
   ```
2. **Set up a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app:**
   ```bash
   python app.py
   ```
5. **Open your browser:**
   Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 🤖 Model Info
- The machine learning model (`fclass_prediction_model.pkl`) was **trained separately** using the classic Iris dataset and scikit-learn’s Logistic Regression.
- The model is **loaded at runtime** for fast, accurate predictions.
- The app uses a `LabelEncoder` to map model outputs to human-readable species names.

## 📦 Project Structure
```
backend_workshop/
├── app.py                  # Flask backend and prediction logic
├── fclass_prediction_model.pkl  # Pre-trained ML model (not included in repo)
├── static/
│   ├── flower.jpg          # Background image
│   ├── iris-sesota.jpg     # Setosa image
│   ├── iris-versicolor.jpg # Versicolor image
│   └── iris-viginca.jpg    # Virginica image
├── templates/
│   ├── index.html          # Main form page
│   └── results.html        # Results/info page
└── README.md
```

## 🙏 Credits
- **Images:** Your own local images, plus Unsplash/Wikimedia for inspiration
- **ML Model:** Trained using scikit-learn and the Iris dataset
- **UI/UX:** Designed for clarity, accessibility, and engagement

## 📚 Further Improvements
- Add user authentication
- Track prediction history
- Deploy to the cloud (Heroku, Vercel, etc.)

---

**Built with ❤️ by [holly-agyei](https://github.com/holly-agyei)**