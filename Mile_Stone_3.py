import streamlit as st
import os
from pypdf import PdfReader
import cv2
import google.generativeai as genai
from PIL import Image
from sqlalchemy import create_engine, text
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# User credentials
USER_CREDENTIALS = {"username": "sai", "password": "123"}
UPLOAD_DIRECTORY = "uploaded_files"

# Ensure directories exist
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs('final_check_images', exist_ok=True)


# Login page
def login():
    st.title("Login Page")
    username = st.text_input("Enter your Username")
    password = st.text_input('Enter Your Password', type="password")
    login_button = st.button("Login Now")

    if login_button:
        if not username or not password:
            st.error("Both username and password are required!")
        elif username == USER_CREDENTIALS["username"] and password == USER_CREDENTIALS["password"]:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome {username}! You have logged in successfully.")
        else:
            st.error("Invalid username or password. Please try again.")


# Database connection function
def create_connection():
    db_user = "postgres"  # Update with your username
    db_password = "123"  # Update with your password
    db_host = "localhost"
    db_port = "5432"
    db_name = "Dataextraction"
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(connection_string)
    return engine



def process_and_save_checks(file_length, Py_read_obj):
    for i in range(file_length):
        for id, image in enumerate(Py_read_obj.pages[i].images):
            image_data = image.data
            temp_image_path = f'temp_image_{id}.jpg'
            with open(temp_image_path, 'wb') as f:
                f.write(image_data)

            img = cv2.imread(temp_image_path)
            if img is None:
                print(f"Failed to load image: {temp_image_path}")
                continue

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_img, threshold1=50, threshold2=150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if 1.5 < aspect_ratio < 4.0 and w > 100 and h > 40:
                    valid_contours.append((x, y, w, h))

            if valid_contours:
                valid_contours.sort(key=lambda bbox: bbox[2] * bbox[3], reverse=True)
                x, y, w, h = valid_contours[0]
                padding_height = 800
                padding_width = 2200
                x = max(0, x - padding_width)
                y = max(0, y - padding_height)
                w = min(img.shape[1] - x, w + 2 * padding_width)
                h = min(img.shape[0] - y, h + 2 * padding_height)
                cropped_check = img[y:y + h, x:x + w]
                output_filename = f"check_{i}_{id}.png"
                output_path = os.path.join('final_check_images', output_filename)
                cv2.imwrite(output_path, cropped_check)

            os.remove(temp_image_path)


# PDF upload page
def show_pdf_upload():
    st.title("Upload PDF")
    st.write("Upload a PDF file for processing.")
    pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if pdf_file:
        file_path = os.path.join(UPLOAD_DIRECTORY, pdf_file.name)
        with open(file_path, "wb") as f:
            f.write(pdf_file.read())
        st.success(f"File uploaded and saved as: {file_path}")
        Py_read_obj = PdfReader(file_path)
        file_length = len(Py_read_obj.pages)
        process_and_save_checks(file_length, Py_read_obj)

        # After processing, extract data and store it in the database
        data = ExtractData()
        
        if data:
            st.success("Data extracted successfully!")
            engine = create_connection()
            store_data_in_database(data, engine)

def ExtractData():
    api_key = 'AIzaSyAtb8mTPbhDk1_V3bDBvnLa1HDKBJWXvvw'  # Replace with your actual API key
    if not api_key:
        raise ValueError("API key is missing.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    prompt = """
    You are given a scanned cheque. Extract its content in JSON format:
    {
        "payee_name": "",
        "cheque_date": "",
        "bank_account_number": "",
        "bank_name": "",
        "amount": "",
        "ifsc_code": ""
    }
    """

    def Model(image_path):
        try:
            opened_image = Image.open(image_path)
        except FileNotFoundError:
            raise ValueError(f"Image not found at path: {image_path}")

        response = model.generate_content([prompt, opened_image])
        # Replace 'null' with 'None' to make it valid Python
        sanitized_response = response.text.replace("null", "None").replace("\n", "").replace("```json", "").replace("```", "")
        
        return sanitized_response


    extracted_data = []

    def process_all_images(directory_path):
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check file extensions
                image_path = os.path.join(directory_path, filename)
                model_output = Model(image_path)
                extracted_data.append(eval(model_output))  # Convert JSON string to dictionary

    process_all_images('./final_check_images')
    return extracted_data


def extract_data_from_db():
    try:
        
        # Replace these with your actual database credentials
        DATABASE_URL = "postgresql://postgres:123@localhost:5432/Dataextraction"
        engine = create_engine(DATABASE_URL)

        # Example query to fetch cheque data
        query = """
        SELECT payee_name, cheque_date, bank_account_number, bank_name, amount, ifsc_code
        FROM cheque_data;
        """

        # Read data from database into a pandas DataFrame
        df = pd.read_sql(query, engine)

        # Close the connection
        engine.dispose()

        # Return the data as a list of dictionaries (or any suitable format)
        return df.to_dict(orient='records')
    except Exception as e:
        st.error(f"Error extracting data from database: {e}")
        return None


# Function to display the extracted data and perform analytics
def show_data_extraction(data):
    st.title("Data Extraction and Analytics")
    st.write("Extracted cheque data is displayed below.")

    # Check the structure of extracted data
    if not data:
        st.warning("No data extracted.")
        return

    # Convert extracted data into a DataFrame for better display and analytics
    try:
        df = pd.DataFrame(data)
        
    except Exception as e:
        st.error(f"Error converting data into DataFrame: {e}")
        return

    # Handle case where 'amount' column might be missing
    if 'amount' not in df.columns:
        st.warning("Amount column is missing in the extracted data.")
        return

    # Clean the 'amount' column by removing commas and converting to float
    try:
        df['amount'] = df['amount'].replace({',': ''}, regex=True)  # Remove commas
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')  # Convert to float
    except Exception as e:
        st.error(f"Error cleaning 'amount' column: {e}")
        return

    # Display the data in a table
    st.subheader("Extracted Data Table")
    st.dataframe(df)

    # Analytics Section
    st.subheader("Analytics and Insights")

    # Total Cheques Processed
    total_cheques = len(df)
    st.metric(label="Total Cheques Processed", value=total_cheques)

    # Total Amount Processed
    total_amount = df['amount'].sum()  # Sum the amount after converting to float
    st.metric(label="Total Amount Processed", value=f"₹{total_amount:,.2f}")

    # Average Cheque Amount
    avg_cheque = df['amount'].mean()
    st.metric(label="Average Cheque Amount", value=f"₹{avg_cheque:,.2f}")

    # Maximum and Minimum Cheque Amounts
    max_cheque = df['amount'].max()
    min_cheque = df['amount'].min()
    st.write(f"**Maximum Cheque Amount:** ₹{max_cheque:,.2f}")
    st.write(f"**Minimum Cheque Amount:** ₹{min_cheque:,.2f}")

    # Top 5 Largest Cheques
    st.markdown("### Top 5 Largest Cheques")
    top_cheques = df.nlargest(5, 'amount')
    st.dataframe(top_cheques)

    # Additional Analytics: Correlation of Amount and Bank (Bar chart)
    st.markdown("### Amounts per Bank")
    if 'bank_name' in df.columns:
        bank_amounts = df.groupby('bank_name')['amount'].sum().sort_values(ascending=False)
        st.bar_chart(bank_amounts)
    else:
        st.warning("No bank names found in the data.")

def store_data_in_database(data, engine):
    try:
        with engine.connect() as connection:
            for record in data:
                # Extract values from record with default empty string in case of missing data
                payee_name = record.get("payee_name", "")
                cheque_date = record.get("cheque_date", "")
                bank_name = record.get("bank_name", "")
                amount = record.get("amount", "0")
                bank_account_number = record.get("bank_account_number", "")
                ifsc_code = record.get("ifsc_code", "")

                # Skip if any critical fields are missing
                if not payee_name or not cheque_date or not bank_name or not amount:
                    st.warning(f"Skipping record due to missing fields: {record}")
                    continue

                # Ensure amount is a string before replacing commas
                if isinstance(amount, str):
                    amount = amount.replace(',', '')  # Remove commas
                    try:
                        amount = float(amount)  # Convert to float
                    except ValueError:
                        st.warning(f"Skipping record due to invalid amount: {record}")
                        continue
                else:
                    st.warning(f"Invalid amount type for record: {record}")
                    continue

                # Check if record already exists in the database
                query = text("""
                    SELECT amount FROM cheque_data
                    WHERE payee_name = :payee_name AND cheque_date = :cheque_date AND bank_name = :bank_name
                """)
                result = connection.execute(query, {
                    'payee_name': payee_name,
                    'cheque_date': cheque_date,
                    'bank_name': bank_name
                }).fetchone()

                if result:
                    # If record exists, update the amount
                    current_amount = result[0]  # Access the first element of the tuple (amount)
                    updated_amount = current_amount + amount
                    update_query = text("""
                        UPDATE cheque_data 
                        SET amount = :updated_amount 
                        WHERE payee_name = :payee_name 
                        AND cheque_date = :cheque_date 
                        AND bank_name = :bank_name
                    """)
                    connection.execute(update_query, {
                        'updated_amount': updated_amount,
                        'payee_name': payee_name,
                        'cheque_date': cheque_date,
                        'bank_name': bank_name
                    })
                    st.info(f"Updated record: {payee_name}, {cheque_date}, {bank_name}, new amount: {updated_amount}")
                else:
                    # If record does not exist, insert new record
                    insert_query = text("""
                        INSERT INTO cheque_data (payee_name, cheque_date, bank_account_number, bank_name, amount, ifsc_code)
                        VALUES (:payee_name, :cheque_date, :bank_account_number, :bank_name, :amount, :ifsc_code)
                    """)
                    connection.execute(insert_query, {
                        'payee_name': payee_name,
                        'cheque_date': cheque_date,
                        'bank_account_number': bank_account_number,
                        'bank_name': bank_name,
                        'amount': amount,
                        'ifsc_code': ifsc_code
                    })
                    # Commit the changes
                    connection.commit()  # Ensure the insert is committed to the database
                    st.info(f"Inserted new record: {payee_name}, {cheque_date}, {bank_name}, amount: {amount}")
        
        st.success("Data stored in the database successfully!")

    except Exception as e:
        st.error(f"An error occurred while storing data: {e}")

def show_dashboard():
    st.title("Dashboard - Cheque Processing Platform")
    
    # Project Overview Section
    st.subheader("Project Overview")
    st.write("""
    Welcome to the **Cheque Processing Platform**! This advanced platform automates and simplifies the entire process 
    of cheque data extraction and management. By leveraging modern technologies such as machine learning, image 
    processing, and natural language processing, it converts scanned cheque images into structured data that can 
    be easily stored and analyzed.
    """)

    # Features Section
    st.subheader("Features")
    st.write("""
    - **Automatic Cheque Data Extraction**: Extract payee details, cheque amounts, bank names, and other key information from scanned cheques.
    - **Secure Database Integration**: Store cheque data securely with updates to existing records, ensuring accurate tracking.
    - **Advanced Analytics**: Gain insights into cheque data, including total processed amount, average cheque value, and top cheques.
    - **Image Preprocessing & Cropping**: Efficiently crop and preprocess cheque images for improved OCR accuracy.
    - **PDF Processing**: Upload and extract data from PDF documents containing cheque images.
    """)



def main():
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        login()
    else:
        st.sidebar.title(f"Welcome {st.session_state.username}")
        page = st.sidebar.selectbox("Choose a page", ("Upload PDF", "Show Data Analytics",'Dashboard'))
        if page == "Upload PDF":
            show_pdf_upload()
        elif page == "Show Data Analytics":
            data = extract_data_from_db()
            if data:
                show_data_extraction(data)
        elif page=='Dashboard':
            show_dashboard()


if __name__ == "__main__":
    main()
