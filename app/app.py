"""Hugging Face Space / standalone entry point: `python app/app.py`."""

from resenh.app import build

demo = build()

if __name__ == "__main__":
    demo.launch()
