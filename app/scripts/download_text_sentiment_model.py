"""Download the transcript sentiment checkpoint once for subsequent offline use."""
from huggingface_hub import snapshot_download

from app.services.text_sentiment_service import MODEL_DIR, MODEL_ID


def main():
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, local_dir=str(MODEL_DIR))
    print(f"Cached {MODEL_ID} at {MODEL_DIR}")


if __name__ == "__main__":
    main()
