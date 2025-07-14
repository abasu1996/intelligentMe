import streamlit as st
import pandas as pd
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter
import matplotlib.pyplot as plt
import altair as alt
import time
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from langchain.text_splitter import RecursiveCharacterTextSplitter

import os

from openai import OpenAI
import plotly.express as px
import fitz
import base64



msds_hazard_database = [
    {'name': 'Benzene', 'hazard_type': 'Carcinogen', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Methanol', 'hazard_type': 'Flammable', 'severity': 'Medium', 'hazard_score': 4, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Lead', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Acetone', 'hazard_type': 'Flammable', 'severity': 'Low', 'hazard_score': 3, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Ammonia', 'hazard_type': 'Corrosive', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Arsenic', 'hazard_type': 'Carcinogen', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Chlorine', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Ethylene oxide', 'hazard_type': 'Carcinogen', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Formaldehyde', 'hazard_type': 'Carcinogen', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Hydrochloric acid', 'hazard_type': 'Corrosive', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Hydrogen sulfide', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Mercury', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Nitric acid', 'hazard_type': 'Corrosive', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Phosgene', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Sulfuric acid', 'hazard_type': 'Corrosive', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Toluene', 'hazard_type': 'Flammable', 'severity': 'Medium', 'hazard_score': 4, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Acrolein', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Aniline', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Bromine', 'hazard_type': 'Corrosive', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Cadmium', 'hazard_type': 'Carcinogen', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Carbon tetrachloride', 'hazard_type': 'Carcinogen', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Cyanide', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Dichloromethane', 'hazard_type': 'Carcinogen', 'severity': 'Medium', 'hazard_score': 4, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Diethyl ether', 'hazard_type': 'Flammable', 'severity': 'High', 'hazard_score': 4, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Ethylbenzene', 'hazard_type': 'Toxic', 'severity': 'Medium', 'hazard_score': 4, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Hydrazine', 'hazard_type': 'Carcinogen', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Methyl ethyl ketone', 'hazard_type': 'Flammable', 'severity': 'Medium', 'hazard_score': 3, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Nitrobenzene', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Perchloroethylene', 'hazard_type': 'Carcinogen', 'severity': 'Medium', 'hazard_score': 4, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Phenol', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Sodium hydroxide', 'hazard_type': 'Corrosive', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Styrene', 'hazard_type': 'Flammable', 'severity': 'Medium', 'hazard_score': 4, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Tetrachloroethylene', 'hazard_type': 'Carcinogen', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Trichloroethylene', 'hazard_type': 'Carcinogen', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Xylene', 'hazard_type': 'Flammable', 'severity': 'Medium', 'hazard_score': 3, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Aluminum phosphide', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Ammonium nitrate', 'hazard_type': 'Explosive', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Beryllium', 'hazard_type': 'Carcinogen', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Carbon disulfide', 'hazard_type': 'Flammable', 'severity': 'High', 'hazard_score': 4, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Chromium trioxide', 'hazard_type': 'Corrosive', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Dimethyl sulfate', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Epichlorohydrin', 'hazard_type': 'Carcinogen', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Fluorine', 'hazard_type': 'Corrosive', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Hexane', 'hazard_type': 'Flammable', 'severity': 'Medium', 'hazard_score': 4, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Hydrofluoric acid', 'hazard_type': 'Corrosive', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Isopropyl alcohol', 'hazard_type': 'Flammable', 'severity': 'Medium', 'hazard_score': 3, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Lindane', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Methylene chloride', 'hazard_type': 'Carcinogen', 'severity': 'Medium', 'hazard_score': 4, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Naphthalene', 'hazard_type': 'Carcinogen', 'severity': 'Medium', 'hazard_score': 4, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Nickel carbonyl', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Ozone', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Parathion', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Pentachlorophenol', 'hazard_type': 'Carcinogen', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Phosphorus trichloride', 'hazard_type': 'Corrosive', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Potassium cyanide', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Silver nitrate', 'hazard_type': 'Corrosive', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Sodium azide', 'hazard_type': 'Explosive', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Sodium cyanide', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Tetrahydrofuran', 'hazard_type': 'Flammable', 'severity': 'Medium', 'hazard_score': 3, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Thionyl chloride', 'hazard_type': 'Corrosive', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Titanium tetrachloride', 'hazard_type': 'Corrosive', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Toluene diisocyanate', 'hazard_type': 'Carcinogen', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Uranium hexafluoride', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'REACH', 'country': 'EU'},
    {'name': 'Vinyl chloride', 'hazard_type': 'Carcinogen', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'OSHA', 'country': 'USA'},
    {'name': 'Zinc phosphide', 'hazard_type': 'Toxic', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'EPA', 'country': 'Canada'},
    {'name': 'Allyl Alcoholo', 'hazard_type': 'Flammable', 'severity': 'High', 'hazard_score': 5, 'compliance_authority': 'EPA', 'country': 'Canada'}
]






# vecotizer = CountVectorizer()

#X = vecotizer.fit_transform(msds_hazard_database)
Labels = []

Featurs = []
for item in msds_hazard_database:
    Labels.append(item['name'])
    Featurs.append([item['hazard_type'],item['severity']])
    
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
    
hazard_type_encoder = LabelEncoder()
severity_encoder = LabelEncoder()
# hazard_score_encoder = LabelEncoder()

hazard_iter = [item[0] for item in Featurs]
serverity_iter = [item[1] for item in Featurs]
# haz_score_iter = [item[2] for item in Featurs]

encoded_haz_types = hazard_type_encoder.fit_transform(hazard_iter)
encoded_severity_types = severity_encoder.fit_transform(serverity_iter)
# encoded_haz_score_types = hazard_score_encoder.fit_transform(haz_score_iter)


final_encoded_features = [[encoded_haz_types[i],encoded_severity_types[i]] for i in range(len(msds_hazard_database))]

final_df = pd.DataFrame(final_encoded_features)
final_df.columns = ['Hazard Type','Severity']


##Training Split

#set1
X_train,X_test,y_train,y_test = train_test_split(final_encoded_features,Labels,test_size=0.1,random_state=42)

#set 2
X2_train,X2_test,y2_train,y2_test = train_test_split(final_encoded_features,Labels,test_size=0.2,random_state=28)


## Model initialization

model = LogisticRegression(solver='saga',max_iter=1000)

model.fit(X_train,y_train)


y_pred = model.predict(X_test)
accuracy_model = accuracy_score(y_test,y_pred,normalize=np.bool_())





df2 = pd.DataFrame(msds_hazard_database)

label_encoder = LabelEncoder()
df2['hazard_type_encoded'] = label_encoder.fit_transform(df2['hazard_type'])
df2['severity_encoded'] = label_encoder.fit_transform(df2['severity'])


X = df2[['hazard_score', 'severity_encoded']]  
y = df2['hazard_type_encoded']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred2 = model.predict(X_test)

accurate_random = accuracy_score(y_test,y_pred2)*100


def process_pdf(file):
    pdf_reader = PdfReader(file)
    fitz_doc = fitz.open('Training_SDF_doc.pdf')
    for page_sum in range(fitz_doc.page_count):
        page_instance = fitz_doc[0]
        
        
        st.write(fitz_doc[page_sum])
        search_text = 'Carcinogen'
        text_instance = page_instance.search_for(search_text)
        for inst in text_instance:
            highlight = page_instance.add_highlight_annot(inst)
    
    amend_doc = 'Highlighted_doc.pdf'
    with open(amend_doc, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display =  f"""<embed
    class="pdfobject"
    type="application/pdf"
    title="Embedded PDF"
    src="data:application/pdf;base64,{base64_pdf}"
    style="overflow: auto; width: 100%; height: 100%;">"""
    
    st.markdown(pdf_display,unsafe_allow_html=True)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text



def process_csv(file):
    df = pd.read_csv(file)
    df.drop(df.head(7).index, inplace=True)
    ### PROCESSING THE PDF NAMES###
    
    df.reset_index(inplace=True)
    ##  Renaming the Column ##
    def table_creation(dataSet):
        print(dataSet.columns.get_loc('Unnamed: 1'))
        df.rename(columns={dataSet.columns[dataSet.columns.get_loc('Unnamed: 1')]:'Material ID',
        dataSet.columns[dataSet.columns.get_loc('Unnamed: 3')]:'Description',
        dataSet.columns[dataSet.columns.get_loc('Unnamed: 5')]:'Quantity',
        dataSet.columns[dataSet.columns.get_loc('Unnamed: 6')]:'Unit',
        dataSet.columns[dataSet.columns.get_loc('Unnamed: 7')]:'Unit Cost',
        dataSet.columns[dataSet.columns.get_loc('Unnamed: 8')]:'Cost'
                },inplace=True)
    table_creation(df)
    
    df.rename(columns={df.columns[df.columns.get_loc('Description')]:'Material Name'},inplace=True)
    df.fillna('NoData',inplace=True)
    def drop_unnamed(columnName):
        df.drop(columnName,axis=1,inplace=True)
    drop_unnamed(['index','Unnamed: 0','Unnamed: 4','Bill of Materials','Cost'])
    
    bom_material_name = df['Material Name'].to_list()
    
    st.title("Data Table")




    
    # st.markdown(''':green-background[Model Accuracy :{}]'''.format(accurate_random))
    
    def match_bom_with_hazards(bom_material_name, msds_hazard,threshold=70):
        hazardous_material = []
        for material in bom_material_name:
            for hazard in msds_hazard:
                match_score = fuzz.ratio(material.lower(), hazard['name'].lower())
                if(match_score > threshold):
                    hazardous_material.append({
                        'BOM_material':material,
                        'hazard_material':hazard['name'],
                        'hazard_type':hazard['hazard_type'],
                        'severity':hazard['severity'],
                        'hazard_score':hazard['hazard_score'],
                        'compliance_authority':hazard['compliance_authority'],
                        'country':hazard['country']
                    })
        return hazardous_material
    hazard_analysis = match_bom_with_hazards(bom_material_name,msds_hazard_database)
    
    hazard_df = pd.DataFrame(hazard_analysis)
    st.sidebar.header("Pick you filters")
    hazardType = st.sidebar.multiselect("Pick Hazard Type",hazard_df['hazard_type'].unique())
    if not hazardType:
        hazard_df2 = hazard_df.copy()
    else:
        hazard_df2 = hazard_df[hazard_df['hazard_type'].isin(hazardType)]
    st.warning("Hazardous Materials Detected in BOM:\n",icon="🔥")
    # st.write(hazard_df)
    with st.expander("View Table Data"):
        st.write(hazard_df.style.background_gradient(cmap='Blues'))


      
    def generate_summary():
                
        
        # openai.api_key = os.getenv('OPENAI_API_KEY')

        hazard_analysis = match_bom_with_hazards(bom_material_name,msds_hazard_database)
        df = pd.DataFrame(hazard_analysis)

        client = OpenAI(api_key="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")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Generate detailed Summary for {df} this BOM table and also generate a standard handling procedure for each hazardous materials."},
                {"role": "user", "content": f"Provide alternative for those materials which are catagorized as carcinogenic in this dataframe {df}."}
            ],
            max_completion_tokens=1000,
            temperature=0.7
        )

        return response.choices[0].message.content


    st.title('Graphical Analysis')
    
    col1,col2 = st.columns(2)
    with col1:
        fig5 = px.bar(hazard_df2, x='hazard_material', y='hazard_score', 
              title="Hazard Materials vs Hazard Score",
              labels={'hazard_material': 'Hazard Material', 'hazard_score': 'Hazard Score'},
              color='hazard_score')
        st.plotly_chart(fig5,use_container_width=True)
    with col2:
        hazard_valus_count = hazard_df2['hazard_type'].value_counts()
        fig2 = px.pie(values=hazard_valus_count.values, names=hazard_valus_count.index,title="Distribution of Hazard Types",hole=0.3)
        st.plotly_chart(fig2,use_container_width=True)
    summary_button = st.button("Generate BOM summary")
    alternate_suggestion = st.button('Alternate Suggestion')
    if(summary_button):
        generated_text = generate_summary()
        with st.expander('View Generated Summary'):
            st.write(generated_text)
    regenrate = st.button('Regenerate Summary')
    if regenrate:
        generated_text = generate_summary()
        with st.expander('View Regenerated Summary'):
            st.write(generated_text)

    def generate_hazard_report(hazardous_materials):
        if not hazardous_materials:
            st.write("No Hazard Materials Detected in BOM:\n",icon="✅")
            return
        

            
    hazard_report = generate_hazard_report(hazard_analysis)
    
    
    return df


# Function to download processed data as CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')
   
    
def main():

    st.title(":bar_chart: C.A.R.E. AI")

    st.subheader("Upload your BOM to analyze material specifications")
    
    

    uploaded_file = st.file_uploader("Upload CSV", type=['csv','pdf'])
    
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_extension = file_name.split('.')[-1]
        
        if file_extension == 'csv':
            st.write("Scanning the Document...")
            time.sleep(5)
            st.write(f"Processing CSV file: {file_name}")
            df = process_csv(uploaded_file)
            
            
            



            csv_data = convert_df_to_csv(df)

            def talking_agent():
                user_question = st.text_input("What else do you want to know?")

                if user_question is not None and user_question!="":
                    st.write(f"Your question is:{user_question}")
            
            btn_click = st.button("Want to know more.. Click Here!")
            
            if(btn_click):
                talking_agent()
            
            

        elif file_extension == 'pdf':
            st.write(f"Processing PDF file: {file_name}")
            pdf_text = process_pdf(uploaded_file)
            

            st.text_area("Extracted Text from PDF", pdf_text, height=300)
            

            st.download_button(
                label="Download extracted text",
                data=pdf_text,
                file_name='extracted_text.txt',
                mime='text/plain'
            )
            

if __name__ == '__main__':
    main()