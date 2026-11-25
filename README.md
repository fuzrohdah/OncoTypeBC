# OncoTypeBC

Predicts breast cancer subtype (Basal, Her2, LumA, LumB) from a small multi-omic panel, with a Streamlit app.

## Model
- Random Forest (tuned with 5-fold CV), trained on TCGA-BRCA (945 samples)
- 27 features: TPM expression of 20 genes, mutation flags for 5 genes, MYC and CCND1 amplification
- ~94% accuracy on the held-out test set

## Files
- `notebook_models.ipynb` – data prep, training, evaluation
- `app.py` – Streamlit app (upload an `.xlsx`, get subtype and confidence)
- `rf_model.pkl` – trained model
- `testing_sheet.xlsx` – example input

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
The optional LLM explanation needs a Groq API key in `app.py`. The cohort plots need `final_merged_df.csv` (not included).

For research and learning only, not for clinical use.
