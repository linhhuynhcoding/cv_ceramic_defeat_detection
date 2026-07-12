def main():
    print(
        """Ceramic Tile Defect Detection

Run the web app:
  uv run streamlit run app.py

Train model:
  uv run python -m src.train_model

Evaluate:
  uv run python -m src.evaluate
"""
    )


if __name__ == "__main__":
    main()
