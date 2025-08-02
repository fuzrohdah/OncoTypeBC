import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json

st.markdown("<div style='text-align: right'><a href='#faq'><button style='background-color:#4CAF50;color:white;padding:8px 16px;border:none;border-radius:5px;'>FAQ</button></a></div>", unsafe_allow_html=True)


# Load trained model
model = joblib.load(r"C:\Users\haris\OneDrive\Desktop\PROJECT\rf_model.pkl")

#st.title("OncoType: An Integrated Multi-Omic Classifier for Breast Cancer Subtyping")

#blue color heading
st.markdown(
    "<h1 style='color:#1976D2;'>Integrated Multi-Omic Classifier for Breast Cancer Subtyping</h1>", 
    unsafe_allow_html=True
)


st.markdown(""" **A Streamlit app for classifying breast cancer (BRCA) subtypes**: **Basal**, **Her2**, **LumA**, and **LumB**.

* Built using a machine learning (Random Forest) model trained on **TCGA-BRCA multi-omic data** (945 samples).
* Accepts an Excel file as input containing:
    1) **TPM expression** of 20 selected genes
    2) Binary **mutation** status of 5 key genes
    3) **Amplification (CNV)** status of 2 genes: MYC and CCND1
* Input sheet must exactly match the training format (feature order, names, types).
* Internally uses a **hyperparameter-tuned Random Forest** classifier.
* Output shows **predicted subtype** and **prediction confidence**.

**Note**:
* Not clinically validated — intended only for research and educational use.
* Should not guide clinical or therapeutic decisions.
* Interpretability is limited; predictions reflect statistical associations, not causal relationships.
""")






#Upload Excel file
st.subheader("Upload the Excel Sheet with 20 + 5 + 2 features in order")
uploaded_file = st.file_uploader("**Upload Excel file here**", type=["xlsx"])

def explain_prediction_with_groq(features_dict, predicted_class, confidence):
    prompt = f"""Given this BRCA sample with prediction '{predicted_class}' at confidence {confidence:.2f}, and gene-level features: {features_dict}. Explain briefly (in 3-4 lines) why the confidence is high or low. Also provide any biological insight if possible. Finally, include this sentence at the end: 'This explanation is AI-generated using LLMs and is not clinically validated.'"""

    headers = {
        "Authorization": "Bearer <groq_api_key_here>",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Explanation failed. Error {response.status_code}: {response.text}"

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        preds = model.predict(df)
        probs = model.predict_proba(df)
        confidences = probs.max(axis=1) * 100

        df['Predicted_Subtype'] = preds
        df['Confidence (%)'] = confidences.round(2)

        st.success("Prediction completed!")
        st.dataframe(df)

        # Add Groq explanation for first row
        first_sample = df.iloc[0].drop(['Predicted_Subtype', 'Confidence (%)']).to_dict()
        predicted_class = df.iloc[0]['Predicted_Subtype']
        confidence = df.iloc[0]['Confidence (%)']
        explanation = explain_prediction_with_groq(first_sample, predicted_class, confidence)

        st.markdown("**Prediction Explanation (via Groq LLM):**")
        st.info(explanation)

        # Download
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error! Cannot Predict!: {str(e)}")


#pipeline image:
st.markdown("**Project Workflow:** ")


st.image(r"C:\Users\haris\OneDrive\Desktop\PROJECT\cancer.png", caption="Fig. Project Workflow", use_container_width=True)





#Pipeline
st.markdown("""
            
1. **Data Collection**: Retrieved gene expression data (in TPM - Transcripts Per Million), mutation data, and copy number variation (CNV) data for approximately 1,070 BRCA samples from TCGA via cBioPortal.
2. **Initial Cleaning**: Removed duplicate entries, irrelevant columns, and non-primary tumor samples. Sample identifiers were standardized to ensure uniformity across datasets.
3. **Feature Engineering**: Selected 20 genes from the expression dataset using Mutual Information (MI) scores. Additionally, included binary mutation status for 5 key genes and amplification status (CNV) for 2 critical genes—MYC and CCND1.
4. **Data Integration**: Merged the expression, mutation, and CNV datasets using consistent sample IDs. Ensured proper sample overlap and mapped accurate subtype labels.
5. **Final Preprocessing**: Imputed missing values (if any), ensured strict feature ordering, and exported the final dataset. After rigorous preprocessing and filtering, the dataset was reduced to 945 unique, high-quality sample IDs.
6. **ML Model**: Split the dataset into training and testing sets. Trained and fine-tuned a Random Forest classifier using RandomSearchCV, achieving a peak accuracy of **89%** on validation. Model performance was also assessed via confidence scores.
7. **Model Comparison**: Evaluated alternative classifiers including Support Vector Classifier (SVC, 87%), XGBoost (88%), and a basic Artificial Neural Network (validation: 85%). None of these models outperformed the Random Forest in terms of accuracy (89%) or interpretability.
8. **App Development**: Developed a Streamlit web application that accepts an Excel file with preformatted input (identical to the training features) and returns predicted BRCA subtypes with confidence scores. The app also uses a GenAI (groq-api)-based explanation module (LLM) for interpretability insights.
""")







# Load trained data for some insights
merged_df = pd.read_csv(r"C:\Users\haris\OneDrive\Desktop\PROJECT\final_merged_df.csv", sep=",")
# Stats
st.subheader("Exploratory Overview of Training Data")
st.markdown("**BRCA Subtype Distribution:**")
fig1, ax1 = plt.subplots(figsize =(15,10))
sns.countplot(data=merged_df, x='subtype', palette="viridis", ax=ax1)
ax1.set_ylabel("Sample Count")
st.pyplot(fig1)

st.markdown("**CNV Frequencies of Selected Genes:**")
mutation_cnv_cols = [col for col in merged_df.columns if col.endswith("_m") or col.endswith("_amp")]
fig3, ax3 = plt.subplots(figsize = (4,3))
merged_df[mutation_cnv_cols].sum().plot(kind='bar', color='coral', ax=ax3)
ax3.set_ylabel("Positive Samples")
st.pyplot(fig3)








# FAQ
st.markdown("---")
st.markdown("## FAQ", unsafe_allow_html=True)
st.markdown("<div id='faq'></div>", unsafe_allow_html=True)

st.markdown("**Q1. Why didn’t you include BRCA Normal samples?**" )
st.markdown("Due to very few available samples (~20), which is insufficient for learning stable patterns in machine learning models.")

st.markdown("**Q2. Aren’t there simpler and cheaper histopathological methods to classify BRCA?**")
st.markdown("Yes, immunohistochemistry (IHC) and pathology are standard and cost-effective methods to classify BRCA subtypes. However, multi-omic machine learning models offer deeper molecular insights, can uncover hidden patterns beyond visual inspection, and support integrative biomarker discovery. This tool complements, not replaces, traditional methods and is meant solely for exploratory, research-oriented analysis.")
