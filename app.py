from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and feature names
model = joblib.load("model/finance_model.pkl")
feature_names = joblib.load("model/feature_names.pkl")

def generate_tips(category):
    if category == "Overspending":
        return ["You're overspending. Track expenses better.",
                "Cut unnecessary costs.",
                "Try saving at least 10% of your income."]
    elif category == "Balanced":
        return ["You're doing okay!",
                "Set up automated savings.",
                "Plan for long-term goals."]
    elif category == "Saver":
        return ["Excellent savings habit!",
                "Look into mutual funds/FDs.",
                "Keep building your emergency fund."]
    return ["Error generating advice."]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    tips = []
    chart_data = {}
    input_data = {}

    if request.method == "POST":
        try:
            input_data = {}
            for field in feature_names:
                val = request.form.get(field, "")
                if field in ["Occupation", "City_Tier"]:
                    input_data[field] = val
                else:
                    input_data[field] = float(val)

            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            tips = generate_tips(prediction)

            chart_data = {
                "Rent": input_data.get("Rent", 0),
                "Groceries": input_data.get("Groceries", 0),
                "Transport": input_data.get("Transport", 0),
                "Entertainment": input_data.get("Entertainment", 0),
                "Utilities": input_data.get("Utilities", 0),
                "Healthcare": input_data.get("Healthcare", 0),
            }

        except Exception as e:
            prediction = "Error"
            tips = [f"Something went wrong: {e}"]

    return render_template("index.html", prediction=prediction, tips=tips,
                           features=feature_names, inputs=input_data, chart_data=chart_data)

if __name__ == "__main__":
    app.run(debug=True)
