import os
from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf


# =========================================================
# FLASK APP SETUP
# =========================================================

app = Flask(__name__)

app.secret_key = os.environ.get("SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("SECRET_KEY environment variable is not set.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# DATABASE SETUP
# =========================================================

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + os.path.join(BASE_DIR, "foodscanner.db")
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# =========================================================
# UPLOAD FOLDER
# =========================================================

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# USER DATABASE MODEL
# =========================================================

class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(100), unique=True, nullable=False)

    password = db.Column(db.String(200), nullable=False)


# Create database
with app.app_context():
    db.create_all()


# =========================================================
# LOAD FOOD MODEL
# =========================================================

MODEL_PATH = os.path.join(BASE_DIR, "food_classifier_v4.keras")

model = None

if os.path.exists(MODEL_PATH):

    try:

        model = tf.keras.models.load_model(MODEL_PATH)

        print("Food model loaded successfully.")

    except Exception as e:

        print("ERROR loading food model:")
        print(e)

else:

    print("WARNING: food_classifier.keras was not found.")


# =========================================================
# FOOD CLASS NAMES
# IMPORTANT:
# These MUST match the training class order
# =========================================================

CLASS_NAMES = [
    "bisibelebath",
    "chappati",
    "dosa",
    "idly",
    "jamun",
    "momos",
    "pakoda",
    "poori",
    "upma"
]

if model is not None:
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Model predicts {model.output_shape[-1]} classes, "
            f"but CLASS_NAMES contains {len(CLASS_NAMES)} names."
        )


# =========================================================
# FOOD NUTRITION INFORMATION
# Approximate values per serving
# =========================================================

# Standardized nutrition estimates for Food Scanner.
# Values are estimates for the specified serving size.
# Actual nutrition varies with ingredients, portion weight, and cooking method.

FOOD_NUTRITION = {
    "bisibelebath": {
        "serving_size": "100 g cooked",
        "calories": 150,
        "protein": 4.0,
        "carbs": 25.0,
        "fiber": 2.5
    },

    "chappati": {
        "serving_size": "2 medium chapatis",
        "calories": 224,
        "protein": 5.8,
        "carbs": 35.8,
        "fiber": 6.4
    },

    "dosa": {
        "serving_size": "1 medium plain dosa",
        "calories": 138,
        "protein": 3.7,
        "carbs": 23.1,
        "fiber": 1.9
    },

    "idly": {
        "serving_size": "4 medium idlis",
        "calories": 260,
        "protein": 9.2,
        "carbs": 56.4,
        "fiber": 1.2
    },

    "jamun": {
        "serving_size": "1 medium gulab jamun",
        "calories": 150,
        "protein": 2.0,
        "carbs": 25.0,
        "fiber": 0.5
    },

    "momos": {
        "serving_size": "4 steamed vegetable momos",
        "calories": 160,
        "protein": 5.0,
        "carbs": 26.0,
        "fiber": 2.5
    },

    "pakoda": {
        "serving_size": "1 medium pakoda",
        "calories": 70,
        "protein": 1.5,
        "carbs": 7.0,
        "fiber": 0.8
    },

    "poori": {
        "serving_size": "2 medium pooris",
        "calories": 200,
        "protein": 4.0,
        "carbs": 28.0,
        "fiber": 2.0
    },

    "upma": {
        "serving_size": "100 g cooked",
        "calories": 110,
        "protein": 3.0,
        "carbs": 17.0,
        "fiber": 2.0
    }
}
# =========================================================
# LOGIN CHECK
# =========================================================

def login_required():

    return "username" in session


# =========================================================
# FOOD PREDICTION FUNCTION
# =========================================================
def predict_food(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((180, 180))

        image_array = np.array(image).astype("float32")
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model(image_array, training=False).numpy()[0] 

        # Find the two highest prediction scores
        sorted_indices = np.argsort(predictions)[::-1]

        predicted_index = int(sorted_indices[0])
        second_index = int(sorted_indices[1])

        confidence = float(predictions[predicted_index]) * 100
        second_confidence = float(predictions[second_index]) * 100

        prediction_margin = confidence - second_confidence

        # Food recognition rules
        CONFIDENCE_THRESHOLD = 85
        MARGIN_THRESHOLD = 30

        if (
            confidence < CONFIDENCE_THRESHOLD
            or prediction_margin < MARGIN_THRESHOLD
        ):
            return None, confidence, predictions

        predicted_food = CLASS_NAMES[predicted_index]

        return predicted_food, confidence, predictions

    except Exception as e:
        print("Prediction error:", e)
        return None, 0, None


# =========================================================
# HOME / DASHBOARD
# =========================================================

@app.route("/")
def home():

    logged_in = "username" in session

    username = session.get("username")

    return render_template(
        "index.html",
        logged_in=logged_in,
        username=username
    )


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:

            return render_template(
                "register.html",
                error="Please enter username and password."
            )

        existing_user = User.query.filter_by(
            username=username
        ).first()

        if existing_user:

            return render_template(
                "register.html",
                error="Username already exists."
            )

        hashed_password = generate_password_hash(password)

        new_user = User(
            username=username,
            password=hashed_password
        )

        db.session.add(new_user)

        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(
            username=username
        ).first()

        if user and check_password_hash(
            user.password,
            password
        ):

            session["username"] = user.username

            return redirect(url_for("home"))

        return render_template(
            "login.html",
            error="Invalid username or password."
        )

    return render_template("login.html")


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("home"))


# =========================================================
# FOOD SCANNER
# =========================================================


@app.route("/scanner", methods=["GET", "POST"])
def scanner():

    # Login required
    if not login_required():
        return redirect(url_for("login"))

    if request.method == "POST":

        # Check whether an image was uploaded
        if "food_image" not in request.files:
            return render_template(
                "food_scanner.html",
                error="Please select an image.",
                prediction=None,
                confidence=None,
                nutrition=None,
                image_url=None
            )

        file = request.files["food_image"]

        if file.filename == "":
            return render_template(
                "food_scanner.html",
                error="Please select an image.",
                prediction=None,
                confidence=None,
                nutrition=None,
                image_url=None
            )

        # Save uploaded image
        filename = secure_filename(file.filename)

        if not filename:
            return render_template(
                "food_scanner.html",
                error="Invalid filename. Please select another image.",
                prediction=None,
                confidence=None,
                nutrition=None,
                image_url=None
            )

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(image_path)

        # Predict food
        # Predict food
        print("DEBUG: Starting food prediction...", flush=True)

        prediction, confidence, probabilities = predict_food(image_path)

        print("DEBUG: Food prediction finished.", flush=True)

        # If prediction failed
        if prediction is None:
            return render_template(
                "food_scanner.html",
                error=(
                    "Food Not Recognized. Please upload a clearer "
                    "image of one of our supported foods."
                ),
                prediction=None,
                confidence=None,
                nutrition=None,
                image_url=None
            )

        # Get nutrition information
        nutrition = FOOD_NUTRITION.get(prediction)

        # Image URL
        image_url = url_for(
            "static",
            filename=f"uploads/{filename}"
        )

        # Display prediction and nutrition
        return render_template(
            "food_scanner.html",
            prediction=prediction,
            confidence=round(float(confidence), 2),
            nutrition=nutrition,
            image_url=image_url,
            error=None
        )

    # Display scanner page before an image is uploaded
    return render_template(
        "food_scanner.html",
        prediction=None,
        confidence=None,
        nutrition=None,
        image_url=None,
        error=None
    )



# =========================================================
# ALTERNATIVE FOOD SCANNER URL
# =========================================================

@app.route("/food-scanner")
def food_scanner():

    return redirect(url_for("scanner"))


# =========================================================
# BMI CALCULATOR
# =========================================================

@app.route("/bmi", methods=["GET", "POST"])
def bmi():

    if not login_required():

        return redirect(url_for("login"))

    bmi_value = None
    category = None
    suggestion = None

    if request.method == "POST":

        try:

            height = float(
                request.form.get("height")
            )

            weight = float(
                request.form.get("weight")
            )

            # Height entered in feet
            height_meters = height * 0.3048

            if height_meters <= 0 or weight <= 0:

                raise ValueError

            bmi_value = weight / (
                height_meters ** 2
            )

            bmi_value = round(bmi_value, 2)

            # BMI category
            if bmi_value < 18.5:

                category = "Underweight"

                suggestion = (
                    "Focus on regular, nutrient-dense meals "
                    "with enough protein, healthy fats, fruits, "
                    "vegetables and whole grains. Avoid skipping meals."
                )

            elif bmi_value < 25:

                category = "Normal weight"

                suggestion = (
                    "Maintain a balanced diet with vegetables, "
                    "fruits, whole grains and adequate protein. "
                    "Stay physically active and maintain good sleep habits."
                )

            elif bmi_value < 30:

                category = "Overweight"

                suggestion = (
                    "Focus on gradual, sustainable changes such as "
                    "more vegetables, fiber and protein, regular physical "
                    "activity and limiting highly processed foods. "
                    "Avoid crash diets."
                )

            else:

                category = "Obese"

                suggestion = (
                    "Focus on sustainable nutrition and regular physical "
                    "activity rather than rapid weight loss. Consider "
                    "speaking with a qualified healthcare professional "
                    "for personalized guidance."
                )

        except (ValueError, TypeError):

            return render_template(
                "bmi.html",
                error="Please enter valid height and weight."
            )

    return render_template(
        "bmi.html",

        bmi=bmi_value,

        category=category,

        suggestion=suggestion
    )


# =========================================================
# PROTEIN FOODS
# =========================================================

@app.route("/protein-foods")
def protein_foods():
    if not login_required():
        return redirect(url_for("login"))

    return render_template("protein.html")

# =========================================================
# HEALTHY CHOICES
# =========================================================

@app.route("/healthy-choices")
def healthy_choices():

    if not login_required():

        return redirect(url_for("login"))

    return render_template(
        "healthy_choices.html"
    )
@app.route("/sparks")
def sparks():

    if not login_required():
        return redirect(url_for("login"))

    return render_template("sparks.html")


# =========================================================
# ABOUT US
# =========================================================

@app.route("/about")
def about():
    return render_template("about.html")

# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )