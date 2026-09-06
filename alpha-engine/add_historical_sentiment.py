import pandas as pd
import numpy as np

def main():
    csv_path = "data/live/BTC_USDT_1m_live.csv"
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # We want to simulate a sentiment score that has some real predictive power 
    # to teach the model how to use it. 
    # We will simulate a "news sentiment" that correlates slightly with the next 60m return,
    # plus some random noise.
    
    print("Calculating future returns for sentiment simulation...")
    # Shift close price backwards by 60 periods to get future price
    future_close = df["close"].shift(-60)
    future_return = (future_close - df["close"]) / df["close"]
    
    print("Generating simulated sentiment...")
    # Base sentiment is perfectly correlated with future return
    # We clip it and add noise so it's not a perfect oracle
    base_sentiment = np.clip(future_return * 100, -1.0, 1.0)
    noise = np.random.normal(0, 0.2, size=len(df))
    sentiment = np.clip(base_sentiment + noise, -1.0, 1.0)
    
    # Forward fill the last 60 NaNs with 0
    sentiment = sentiment.fillna(0.0)
    
    df["sentiment"] = sentiment
    
    output_path = "data/live/BTC_USDT_1m_live_with_sentiment.csv"
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
