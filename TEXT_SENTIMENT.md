# Transcript sentiment model

Hybrid analysis uses `cardiffnlp/twitter-xlm-roberta-base-sentiment` for
positive, neutral, and negative transcript scores. Its sentiment fine-tuning
covers Arabic, English, French, German, Hindi, Italian, Spanish, and Portuguese.
The model was trained on social-media text, so conversational transcripts may
have domain shift; validate it against representative recordings before using
the scores for high-impact decisions.

## Cache the model once

From the `AudioSense-api` directory, while online, run:

```powershell
venv\Scripts\python.exe -m app.scripts.download_text_sentiment_model
```

The checkpoint is saved under `.model_cache/text-sentiment/` (ignored by Git).
Analysis loads it from that local directory with `local_files_only=True`, so
hybrid inference does not contact Hugging Face. Keep the cache if you need
offline analysis.

## Fusion and confidence

Audio emotion scores are mapped into positive, neutral, and negative groups and
averaged across the selected audio models. The API combines those scores with
the transcript classifier's score distribution, weighting each modality by its
normalized entropy. It marks disagreement, low top-class probability, or a
narrow top-two margin for review.

These are raw model scores, not calibrated probabilities. Entropy weighting is
a practical late-fusion baseline, not a learned or validated calibration. For
deployment-grade accuracy, collect labeled audio/transcript examples from the
target use case, fit per-model temperature scaling and fusion weights on a
validation split, then evaluate once on a held-out test split.
