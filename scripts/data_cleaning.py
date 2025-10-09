import pandas as pd

def clean_data(input_path: str, output_path: str):
    """
    Clean the raw subsidy claim data.
    """
    df = pd.read_csv(input_path)

    # Example cleaning logic
    df.drop_duplicates(subset=['Name', 'Aadhaar', 'SubsidyType', 'Date'], inplace=True)
    df.fillna({'ClaimAmount': 0}, inplace=True)

    # Optional: add flags for duplicate Aadhaar/claims
    df['AadhaarFlag'] = df.duplicated(subset=['Aadhaar'], keep=False)
    df['ClaimFlag'] = df.duplicated(subset=['Name', 'SubsidyType'], keep=False)
    df['FraudFlag'] = False  # placeholder for extra fraud detection

    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to: {output_path}")
