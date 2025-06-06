import pandas as pd

class AnimeDataLoader:
    def __init__(self, original_csv: str, processed_csv: str):
        # Initialize the class with the paths of input and output CSV files
        self.original_csv = original_csv
        self.processed_csv = processed_csv

    def load_and_process(self):
        # Read the original CSV file with UTF-8 encoding
        # error_bad_lines=False will skip any bad lines silently (deprecated warning, but kept as you wrote)
        # Then drop rows that have any missing (NaN) values
        df = pd.read_csv(self.original_csv, encoding='utf-8', error_bad_lines=False).dropna()

        # Define the required columns to check if they exist in the CSV
        required_cols = {'Name', 'sypnopsis', 'Genres'}
        # Find which required columns are missing in the CSV columns
        missing = required_cols - set(df.columns)
        # If any required columns are missing, raise an error and stop execution
        if missing:
            raise ValueError(f"Missing columns in CSV file: {', '.join(missing)}")

        # Create a new column 'combined_info' by concatenating 'Name', 'sypnopsis', and 'Genres' with labels
        df['combined_info'] = (
            "Title: " + df['Name'] + ". Overview: " + df['sypnopsis'] + " Genres: " + df['Genres']
        )

        # Save only the new 'combined_info' column to a new CSV file with UTF-8 encoding, without row index
        df[['combined_info']].to_csv(self.processed_csv, index=False, encoding="utf-8")
        
        # Return the path of the processed CSV file
        return self.processed_csv
