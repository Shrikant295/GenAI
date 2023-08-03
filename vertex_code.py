import vertexai, streamlit as st
import pandas as pd

from vertexai.preview.language_models import CodeGenerationModel
from vertexai.preview.language_models import TextGenerationModel


PROJECT_ID = 'reflected-radio-394516' # @param {type:"string"}

LOCATION = 'us-central1'  # @param {type:"string"}

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Streamlit app


st.title('AI Shark Assitance')

# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#   df = pd.read_csv(uploaded_file)
#   st.write(df)

slider_code = ["Data Ingestion","Data Quality Check", "EDA And Data Cleaning", "Model Prediction and Evaluation", "Model Enhancement","Unit Test(Code)"]
slider_text = ["Unit Test (Text)","Documents/guideline"]

slider_var = st.sidebar.selectbox(
    "Task bar",
    (slider_code + slider_text))

#print(slider_var)
# with open('style.css')as f:
    
prompt = st.text_area('Input query')


if st.button("Submit"):
    if not prompt.strip():
        st.write(f"Please submit your query.")
    else:
        if slider_var in slider_code:
            try:
                model = CodeGenerationModel.from_pretrained("code-bison@001")
                response = model.predict(
                    prompt,
                    temperature=0.2,
                    max_output_tokens=1256
                )

                st.success(response)
                st.download_button('Download Results', response.text,file_name='Result.py')

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            try:
                model = TextGenerationModel.from_pretrained("text-bison@001")
                response = model.predict(
                    prompt,
                    temperature=0.1,
                    max_output_tokens=1024,
                    top_p = 0.8,
                    top_k = 40
                )
                st.success(response)
                st.download_button('Download Results', response.text,file_name='Result.txt')
                           
            except Exception as e:
                st.error(f"An error occurred: {e}")
