from flask import Flask, render_template, request
import whisper
from textblob import TextBlob
import os

app = Flask(__name__)

model = whisper.load_model("base")

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():

    text = ""
    sentiment = ""

    if request.method == "POST":

        audio = request.files["audio"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], audio.filename)
        audio.save(path)

        result = model.transcribe(path)
        text = result["text"]

        analysis = TextBlob(text)

        if analysis.sentiment.polarity > 0:
            sentiment = "Positive 😊"
        elif analysis.sentiment.polarity < 0:
            sentiment = "Negative 😔"
        else:
            sentiment = "Neutral 😐"

    return render_template("index.html", text=text, sentiment=sentiment)


if __name__ == "__main__":
    app.run(debug=True)