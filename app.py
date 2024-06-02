import pandas as pd
import numpy as np
from transformers import pipeline
import streamlit as st
import io
import sys

# Function to redirect print statements to Streamlit
class StreamlitLogger(io.StringIO):
    def __init__(self):
        super().__init__()
        self.output_area = st.empty()

    def write(self, s):
        super().write(s)
        self.output_area.text(self.getvalue())

# Set up the Streamlit logger
streamlit_logger = StreamlitLogger()
sys.stdout = streamlit_logger

# Streamlit interface for uploading the Excel file and displaying results
st.title("Suspicious Transaction Detection")

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    # Load the transaction data
    print("Loading the transaction data...")
    df_transactions = pd.read_excel(uploaded_file, sheet_name='Transactions')
    df_account_details = pd.read_excel(uploaded_file, sheet_name='AccountDetails')
    print("Data loaded successfully.\n")

    # Ensure Amount column contains only numerical values and no missing values
    print("Cleaning Amount column...")
    df_transactions['Amount'] = pd.to_numeric(df_transactions['Amount'], errors='coerce')
    df_transactions = df_transactions.dropna(subset=['Amount'])
    print("Amount column cleaned.\n")

    # Print out some details to debug
    print(f"First few rows of transactions data:\n{df_transactions.head()}")
    print(f"Statistics of Amount column:\n{df_transactions['Amount'].describe()}")

    # Check the keys used for merging
    print(f"Unique SenderAccounts in transactions data: {df_transactions['SenderAccount'].nunique()}")
    print(f"Unique AccountIDs in account details data: {df_account_details['AccountID'].nunique()}")
    print(f"First few SenderAccounts in transactions data:\n{df_transactions['SenderAccount'].head()}")
    print(f"First few AccountIDs in account details data:\n{df_account_details['AccountID'].head()}")

    # Merge the transactions with account details to get the account holder's country
    print("Merging transaction data with account details...")
    df_merged = pd.merge(df_transactions, df_account_details, left_on='SenderAccount', right_on='AccountID', suffixes=('', '_Sender'))
    print("Data merged successfully.\n")

    # Ensure Amount column in merged DataFrame is correct
    print(f"First few rows of merged data:\n{df_merged.head()}")
    print(f"Statistics of Amount column in merged data:\n{df_merged['Amount'].describe()}")

    # Criteria for identifying suspicious transactions
    print("Identifying potentially suspicious transactions...")

    # High Amount
    amount_threshold = df_merged['Amount'].mean() + 2 * df_merged['Amount'].std()  # More flexible threshold
    print(f"High amount threshold: {amount_threshold}")
    df_merged['HighAmount'] = df_merged['Amount'] > amount_threshold

    # International Transactions
    high_risk_countries = ['Somalia', 'Syria', 'North Korea', 'Iran', 'Afghanistan', 'Yemen', 'Iraq', 'Libya', 'Sudan', 'South Sudan', 'Venezuela', 'Myanmar', 'Zimbabwe', 'Cuba', 'Eritrea', 'Burundi', 'Haiti', 'Central African Republic', 'DR Congo', 'Pakistan']
    df_merged['InternationalTransaction'] = df_merged['Country'].isin(high_risk_countries)

    # Frequent Transactions
    df_merged['TransactionDate'] = pd.to_datetime(df_merged['Date'])
    df_merged = df_merged.sort_values(by=['SenderAccount', 'TransactionDate'])
    df_merged['PreviousTransactionDate'] = df_merged.groupby('SenderAccount')['TransactionDate'].shift(1)
    df_merged['TimeDifference'] = (df_merged['TransactionDate'] - df_merged['PreviousTransactionDate']).dt.total_seconds() / 3600
    df_merged['FrequentTransaction'] = df_merged['TimeDifference'] < 6  # Broaden the range for frequent transactions

    # Mark as suspicious if any criteria are met
    df_merged['Suspicious'] = df_merged[['HighAmount', 'InternationalTransaction', 'FrequentTransaction']].any(axis=1)

    # Filter suspicious transactions
    df_suspicious = df_merged[df_merged['Suspicious']]
    print(f"Number of suspicious transactions identified: {len(df_suspicious)}\n")

    # Check the first few suspicious transactions
    if len(df_suspicious) > 0:
        print("First few suspicious transactions:")
        print(df_suspicious.head())
    else:
        print("No suspicious transactions found.\n")

    # Load a pre-trained GPT-2 pipeline
    print("Loading a pre-trained GPT-2 pipeline...")
    llm = pipeline('text-generation', model='gpt2')
    print("GPT-2 pipeline loaded.\n")

    # Function to generate explanations
    def generate_explanation(transaction_details):
        # Create a concise prompt with relevant details
        prompt = (
            f"Transaction ID: {transaction_details['TransactionID']}\n"
            f"Date: {transaction_details['Date']}\n"
            f"Sender Account: {transaction_details['SenderAccount']}\n"
            f"Receiver Account: {transaction_details['ReceiverAccount']}\n"
            f"Amount: {transaction_details['Amount']}\n"
            f"Currency: {transaction_details['Currency']}\n"
            f"Transaction Type: {transaction_details['TransactionType']}\n"
            f"Country: {transaction_details['Country']}\n"
            f"Status: {transaction_details['Status']}\n"
            f"Reason for suspicion: This transaction involves a high amount, an international transaction, or frequent transactions.\n"
            f"Explain why this transaction might be considered suspicious."
        )
        explanation = llm(prompt, max_new_tokens=50, truncation=True)
        return explanation[0]['generated_text']

    # Generate explanations for suspicious transactions if any are found
    explanations = []
    if len(df_suspicious) > 0:
        print("Generating explanations for suspicious transactions...")
        for idx, row in df_suspicious.iterrows():
            transaction_details = row.to_dict()
            explanation = generate_explanation(transaction_details)
            explanations.append({
                'TransactionID': row['TransactionID'],
                'Explanation': explanation
            })
            print(f"Transaction {row['TransactionID']}: {explanation}\n")

    # Merge explanations back to the main dataframe
    if len(explanations) > 0:
        df_explanations = pd.DataFrame(explanations)
        df_merged = pd.merge(df_merged, df_explanations, on='TransactionID', how='left')

    # Display the complete dataframe with suspicious flag and explanations
    st.write(df_merged)

    # Option to download the final dataframe
    @st.cache
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df_merged)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='complete_transactions_with_suspicious_flags_and_explanations.csv',
        mime='text/csv',
    )
