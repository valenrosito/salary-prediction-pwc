import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.persistence import load_object
from src.config.settings import MODEL_DIR

st.set_page_config(
    page_title="Predictor de Salarios",
    page_icon="💰",
    layout="wide"
)

@st.cache_resource
def load_model_and_pipeline():
    try:
        model_path = Path(MODEL_DIR) / "predictor_model.joblib"
        pipeline_path = Path(MODEL_DIR) / "preprocessor_pipeline.joblib"
        
        model = load_object(model_path)
        preprocessing_pipeline = load_object(pipeline_path)
        
        return model, preprocessing_pipeline
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None


def main():
    st.title("💰 Predictor de Salarios")
    st.markdown("### Sistema de predicción basado en Machine Learning")
    
    model, pipeline = load_model_and_pipeline()
    
    if model is None:
        st.error("No se pudo cargar el modelo. Verifica que exista el archivo del modelo.")
        return
    
    st.success("✓ Modelo cargado correctamente")
    
    tab1, tab2 = st.tabs(["📊 Predicción Individual", "📁 Predicción por Lote (CSV)"])
    
    with tab1:
        st.markdown("#### Ingresa los datos para predecir el salario")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Edad", min_value=18, max_value=100, value=30, step=1)
            gender = st.selectbox("Género", ["Male", "Female"])
            education_level = st.selectbox(
                "Nivel Educativo", 
                ["Bachelor's", "Master's", "PhD"]
            )
        
        with col2:
            years_experience = st.number_input(
                "Años de Experiencia", 
                min_value=0.0, 
                max_value=50.0, 
                value=5.0, 
                step=0.5
            )
            job_title = st.text_input("Título del Trabajo", value="Software Engineer")
        
        description = st.text_area(
            "Descripción del Perfil",
            value="Experienced professional with strong technical skills",
            height=100
        )
        
        if st.button("🔮 Predecir Salario", type="primary"):
            try:
                input_data = pd.DataFrame([{
                    "Age": age,
                    "Gender": gender,
                    "Education Level": education_level,
                    "Job Title": job_title,
                    "Years of Experience": years_experience,
                    "Description": description
                }])
                
                prediction = model.predict(input_data)
                predicted_salary = float(prediction[0])
                
                st.markdown("---")
                st.markdown("### 📈 Resultado de la Predicción")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Salario Predicho", f"${predicted_salary:,.2f}")
                
                with col2:
                    monthly_salary = predicted_salary / 12
                    st.metric("Salario Mensual", f"${monthly_salary:,.2f}")
                
                with col3:
                    hourly_rate = predicted_salary / (52 * 40)
                    st.metric("Tarifa por Hora", f"${hourly_rate:.2f}")
                
            except Exception as e:
                st.error(f"Error en la predicción: {str(e)}")
    
    with tab2:
        st.markdown("#### Carga un archivo CSV para predicciones en lote")
        
        st.info("""
        **Formato del CSV requerido:**
        - Age
        - Gender
        - Education Level
        - Job Title
        - Years of Experience
        - Description
        """)
        
        uploaded_file = st.file_uploader(
            "Selecciona un archivo CSV", 
            type=['csv'],
            key='csv_uploader'
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
                
                st.markdown("#### Vista Previa de los Datos")
                st.dataframe(df.head(10), use_container_width=True)
                
                required_columns = [
                    "Age", "Gender", "Education Level", 
                    "Job Title", "Years of Experience", "Description"
                ]
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Columnas faltantes: {', '.join(missing_columns)}")
                else:
                    if st.button("🚀 Generar Predicciones", type="primary"):
                        with st.spinner("Procesando predicciones..."):
                            predictions = model.predict(df[required_columns])
                            df['Predicted_Salary'] = predictions
                            
                            st.success(f"✓ Se generaron {len(predictions)} predicciones")
                            
                            st.markdown("#### Resultados con Predicciones")
                            st.dataframe(df, use_container_width=True)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Salario Promedio", f"${df['Predicted_Salary'].mean():,.2f}")
                            
                            with col2:
                                st.metric("Salario Mínimo", f"${df['Predicted_Salary'].min():,.2f}")
                            
                            with col3:
                                st.metric("Salario Máximo", f"${df['Predicted_Salary'].max():,.2f}")
                            
                            csv_buffer = io.StringIO()
                            df.to_csv(csv_buffer, index=False)
                            csv_data = csv_buffer.getvalue()
                            
                            st.download_button(
                                label="📥 Descargar Resultados (CSV)",
                                data=csv_data,
                                file_name="predicciones_salarios.csv",
                                mime="text/csv"
                            )
                            
                            st.markdown("#### Distribución de Salarios Predichos")
                            st.bar_chart(df['Predicted_Salary'])
                            
            except Exception as e:
                st.error(f"Error al procesar el archivo: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Predictor de Salarios v1.0 | Modelo: LightGBM | Precisión: R² = 0.89</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
