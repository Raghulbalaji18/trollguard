import tkinter as tk
from tkinter import ttk, messagebox
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download required resources
nltk.download("vader_lexicon", quiet=True)

# Load NLP models
print("Loading AI models... please wait ⏳")
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True)
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
sarcasm_model = pipeline("text-classification", model="mrm8488/t5-base-finetuned-sarcasm-twitter")
vader = SentimentIntensityAnalyzer()
print("✅ Models loaded successfully!")


# --- Analysis Function ---
def analyze_text():
    text = text_input.get("1.0", "end-1c").strip()
    if not text:
        messagebox.showwarning("Input Required", "Please enter text for analysis.")
        return

    # Run all models
    tox_results = toxicity_model(text)[0]
    toxic_score = [r for r in tox_results if r['label'] == 'toxic'][0]['score']

    emo_results = emotion_model(text)[0]
    emotion = sorted(emo_results, key=lambda x: x['score'], reverse=True)[0]

    sarcasm_result = sarcasm_model(text)[0]['label']
    sarcasm_detected = "sarcasm" in sarcasm_result.lower()

    vader_score = vader.polarity_scores(text)
    compound = vader_score["compound"]

    # Determine tone
    if toxic_score > 0.6 or compound < -0.5:
        classification = "⚠️ Toxic / Abusive"
        color = "#ef233c"
    elif sarcasm_detected:
        classification = "😏 Sarcastic"
        color = "#ffb703"
    elif emotion['label'] in ["disgust", "anger"]:
        classification = "🤢 Disgust / Angry"
        color = "#f77f00"
    elif compound >= 0.3:
        classification = "😊 Positive / Normal"
        color = "#06d6a0"
    else:
        classification = "😐 Neutral / Normal"
        color = "#118ab2"

    confidence = (
        max(toxic_score, emotion['score'], abs(compound))
        * (1.2 if sarcasm_detected else 1)
    )
    confidence = min(confidence * 100, 99.9)

    # Update results in UI
    result_label.config(text=classification, fg=color)
    toxicity_label.config(text=f"Toxicity: {toxic_score:.2f}")
    emotion_label.config(text=f"Emotion: {emotion['label'].capitalize()}")
    sarcasm_label.config(text=f"Sarcasm: {'Yes' if sarcasm_detected else 'No'}")
    confidence_label.config(text=f"Confidence: {confidence:.1f}%")

    progress["value"] = confidence


# --- UI Setup ---
root = tk.Tk()
root.title("🤖 TrollGuard 4.0 — AI Tone Analyzer")
root.geometry("700x600")
root.config(bg="#00171f")

# Title
tk.Label(
    root,
    text="🤖 TrollGuard 4.0",
    font=("Arial Black", 22, "bold"),
    fg="#00b4d8",
    bg="#00171f",
).pack(pady=10)

tk.Label(
    root,
    text="AI + NLP Real-Time Tone Analyzer",
    font=("Segoe UI", 12),
    fg="#48cae4",
    bg="#00171f",
).pack()

# Textbox
text_input = tk.Text(root, height=8, width=70, font=("Segoe UI", 11), wrap="word", bg="#edf6f9", fg="#023047")
text_input.pack(pady=20)

# Analyze Button
analyze_btn = tk.Button(
    root,
    text="🔍 Analyze Tone",
    font=("Segoe UI", 12, "bold"),
    bg="#00b4d8",
    fg="white",
    activebackground="#0077b6",
    activeforeground="white",
    relief="flat",
    padx=15,
    pady=8,
    command=analyze_text,
)
analyze_btn.pack(pady=10)

# Results
result_label = tk.Label(root, text="", font=("Arial Black", 16), bg="#00171f")
result_label.pack(pady=15)

toxicity_label = tk.Label(root, text="", font=("Segoe UI", 11), bg="#00171f", fg="white")
toxicity_label.pack()

emotion_label = tk.Label(root, text="", font=("Segoe UI", 11), bg="#00171f", fg="white")
emotion_label.pack()

sarcasm_label = tk.Label(root, text="", font=("Segoe UI", 11), bg="#00171f", fg="white")
sarcasm_label.pack()

confidence_label = tk.Label(root, text="", font=("Segoe UI", 11), bg="#00171f", fg="white")
confidence_label.pack()

# Progress bar
progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress.pack(pady=15)

# Footer
tk.Label(
    root,
    text="© 2025 TrollGuard AI | Built with Transformers + NLP",
    font=("Segoe UI", 9),
    fg="gray",
    bg="#00171f",
).pack(side="bottom", pady=10)

root.mainloop()
