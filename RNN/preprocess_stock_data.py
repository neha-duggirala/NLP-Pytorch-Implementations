import pandas as pd
import matplotlib.pyplot as plt

def process_apple_data(file_path):
    """
    Processes the Apple stock data CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame with Date, Price, and Volume columns.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # 1. Arrange the Date column in ascending order
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    # 2. Remove dollar symbols and convert to float
    df['Close/Last'] = df['Close/Last'].replace({'\$': ''}, regex=True).astype(float)

    # 3. Only use Date, Close/Last, and Volume columns
    df = df[['Date', 'Close/Last', 'Volume']]

    # 4. Rename Close/Last to Price
    df = df.rename(columns={'Close/Last': 'Price'})

    # 5. Return the resulting DataFrame
    return df


def visualize_stock_price(df):
    """
    Visualizes the stock price over time.

    Args:
        df (pd.DataFrame): DataFrame containing Date and Price columns.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Price'], label='Stock Price', color='blue', linewidth=2)
    plt.title('Stock Price Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (in USD)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    file_path = 'RNN/apple_data.csv'
    processed_df = process_apple_data(file_path)
    print(len(processed_df), processed_df.shape)
    visualize_stock_price(processed_df)
