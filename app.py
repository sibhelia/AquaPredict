from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import joblib
import pandas as pd
import numpy as np
import io
import openpyxl

app = Flask(__name__)

FEATURES = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

model  = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/")
def index():
    importances = model.feature_importances_.tolist()
    return render_template("index.html", importances=importances, features=FEATURES)

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory("assets", filename)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    values = [float(data[f]) for f in FEATURES]
    df     = pd.DataFrame([values], columns=FEATURES)
    scaled = scaler.transform(df)
    pred   = int(model.predict(scaled)[0])
    proba  = model.predict_proba(scaled)[0].tolist()
    return jsonify({"prediction": pred, "probability": proba[1]})

@app.route("/bulk", methods=["POST"])
def bulk():
    f  = request.files["file"]
    df = pd.read_csv(f)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return jsonify({"error": f"Eksik sütunlar: {', '.join(missing)}"}), 400

    X      = df[FEATURES]
    scaled = scaler.transform(X)
    preds  = model.predict(scaled)
    probas = model.predict_proba(scaled)[:, 1]

    df["Tahmin"]          = preds
    df["İçilebilir (%)"]  = (probas * 100).round(2)
    df["Durum"]           = pd.Series(preds).map({1: "Güvenli", 0: "Riskli"}).values

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="AquaPredict")
    out.seek(0)

    summary = {
        "safe":  int((preds == 1).sum()),
        "risky": int((preds == 0).sum()),
        "avg":   round(float(probas.mean() * 100), 1),
        "rows":  df.to_dict(orient="records"),
        "cols":  df.columns.tolist(),
    }
    return jsonify(summary)

@app.route("/bulk_excel", methods=["POST"])
def bulk_excel():
    f  = request.files["file"]
    df = pd.read_csv(f)
    X      = df[FEATURES]
    scaled = scaler.transform(X)
    preds  = model.predict(scaled)
    probas = model.predict_proba(scaled)[:, 1]
    df["Tahmin"]         = preds
    df["İçilebilir (%)"] = (probas * 100).round(2)
    df["Durum"]          = pd.Series(preds).map({1:"Güvenli",0:"Riskli"}).values
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="AquaPredict")
    out.seek(0)
    return send_file(out, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     as_attachment=True, download_name="aquapredict_rapor.xlsx")

if __name__ == "__main__":
    app.run(debug=True, port=5000)