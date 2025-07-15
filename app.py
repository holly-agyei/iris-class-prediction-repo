# app.py

from flask import Flask, render_template, request
import pandas as pd
import joblib # For loading the model
from sklearn.preprocessing import LabelEncoder # For converting prediction back to name
import os

app = Flask(__name__)

# --- Load the Machine Learning Model ---
# IMPORTANT: Double-check that 'fclass_prediction_model.pkl' is the EXACT name
# of your saved model file and it's located in the same directory as app.py.
# If your model was saved as 'iris_logistic_model.pkl', you MUST change the filename below!
try:
    model = joblib.load("fclass_prediction_model.pkl")
    print("Machine Learning Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file 'fclass_prediction_model.pkl' not found.")
    print("Please ensure the model file is in the same directory as app.py.")
    # In a production app, you might raise an error or halt execution here.
    model = None # Set model to None to prevent crashes if not found

# --- Initialize LabelEncoder ---
# This maps the numerical predictions (0, 1, 2) back to human-readable species names.
# We fit it with the known possible classes from the Iris dataset.
le = LabelEncoder()
le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
print("LabelEncoder initialized with Iris species.")

# --- Iris Species Info (Open Source Images & Facts) ---
iris_info = {
    'Iris-setosa': {
        'image_url': '/static/iris-sesota.jpg',
        'description': 'Iris setosa, also known as the bristle-pointed iris, is a hardy perennial native to northern North America and eastern Asia.',
        'facts': [
            'Often found in wetlands and along streams.',
            'Its flowers are usually violet-blue, but can also be pink or white.',
            'Setosa means "bristly" in Latin, referring to the plant’s leaves.'
        ],
        'health_benefits': [
            'Traditionally used by indigenous peoples for medicinal purposes (external use only).',
            'Not recommended for internal use due to potential toxicity.'
        ],
        'societal_notes': [
            'A symbol of resilience in cold climates.',
            'Popular in native plant gardens for its beauty and hardiness.'
        ]
    },
    'Iris-versicolor': {
        'image_url': '/static/iris-versicolor.jpg',
        'description': 'Iris versicolor, the blue flag iris, is native to North America and is known for its striking blue-violet flowers.',
        'facts': [
            'Grows in marshes, wet meadows, and along shorelines.',
            'The state flower of Tennessee, USA.',
            'Versicolor means "variously colored" in Latin.'
        ],
        'health_benefits': [
            'Historically used in herbal medicine, but can be toxic if ingested.',
            'Sometimes used in homeopathy (with caution).'
        ],
        'societal_notes': [
            'A favorite in water gardens and natural landscaping.',
            'Featured in Native American folklore and art.'
        ]
    },
    'Iris-virginica': {
        'image_url': '/static/iris-viginca.jpg',
        'description': 'Iris virginica, or the Virginia iris, is a perennial native to eastern North America, thriving in wet habitats.',
        'facts': [
            'Blooms in late spring to early summer.',
            'Flowers are typically blue to violet, sometimes white.',
            'Named after the state of Virginia, USA.'
        ],
        'health_benefits': [
            'Roots were used in traditional medicine, but the plant can be toxic if consumed.',
            'Sometimes planted for erosion control in wetlands.'
        ],
        'societal_notes': [
            'Valued for its role in wetland restoration.',
            'A beautiful addition to rain gardens and naturalized areas.'
        ]
    }
}

# --- Route for the Homepage (Input Form) ---
@app.route('/')
def homepage():
    # This renders the 'index.html' template, which contains the form for user input.
    return render_template('index.html')

# --- Route to Handle Form Submission and Make Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    # 1. Retrieve input values from the form.
    # request.form gets the data submitted via POST.
    # float() converts the string input to a floating-point number.
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Print received inputs to the VS Code terminal for debugging/verification.
    print(f"\n--- Received Inputs from User ---")
    print(f"Sepal Length: {sepal_length}")
    print(f"Sepal Width: {sepal_width}")
    print(f"Petal Length: {petal_length}")
    print(f"Petal Width: {petal_width}")
    print("---------------------------------")

    # 2. Create a single-row Pandas DataFrame from the inputs.
    # This ensures the data is in the exact format (structure and column names)
    # that our machine learning model expects.
    input_data_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                 columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    # Print the DataFrame to the terminal (optional, for further verification).
    print("\n--- Input Data DataFrame for Model ---")
    print(input_data_df)
    print("--------------------------------------")

    # 3. Make the prediction using the loaded model.
    if model: # Only try to predict if the model was loaded successfully
        predict_numeric = model.predict(input_data_df) # Model returns a numerical label (0, 1, or 2).
        
        # Convert the numerical prediction back to the human-readable species name.
        # inverse_transform returns an array, so we take the first element [0].
        predicted_species_name = le.inverse_transform(predict_numeric)[0]
        
        # Print the final prediction to the terminal.
        print(f"Model predicted species: {predicted_species_name}")
    else:
        predicted_species_name = "Prediction Error: Model not loaded."
        print(predicted_species_name)

    # --- Get info for the predicted species ---
    info = iris_info.get(predicted_species_name, None)

    # 4. Render the 'results.html' template.
    # We pass the predicted species name to the template as 'prediction_result'.
    # Ensure 'result.html' uses {{ prediction_result }} to display it.
    return render_template(
        "results.html",
        prediction_result=predicted_species_name,
        iris_info=info
    )

# --- IMPORTANT ---
# Do NOT include app.run() here if you are running your app using `flask run` in the terminal.
# The `flask run` command handles starting the development server for you.
# If you ever run directly with `python app.py`, then you would uncomment:
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)