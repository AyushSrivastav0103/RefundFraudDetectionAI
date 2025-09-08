import numpy as np
import pandas as pd
import datetime
import random

def generate_synthetic_data(n_samples: int = 2000, fraud_ratio: float = 0.15) -> pd.DataFrame:
    """
    Generate synthetic refund claim data for fraud detection.

    Args:
        n_samples (int): Number of total samples to generate.
        fraud_ratio (float): Proportion of fraudulent claims.

    Returns:
        pd.DataFrame: Synthetic dataset.
    """
    np.random.seed(42)
    random.seed(42)

    # --- Customers & Orders ---
    n_customers = int(n_samples * 0.3)  # some customers have multiple orders
    customer_ids = [f"CUST_{i:05d}" for i in range(n_customers)]

    data = {
        "customer_id": np.random.choice(customer_ids, n_samples),
        "order_id": [f"ORDER_{i:06d}" for i in range(n_samples)],
        "claim_id": [f"CLAIM_{i:06d}" for i in range(n_samples)],
        "product_category": np.random.choice(
            ["Electronics", "Groceries", "Clothing", "Home", "Beauty"], n_samples
        ),
        "total_order_value": np.random.uniform(100, 5000, n_samples),
        "refund_amount": np.random.uniform(50, 2000, n_samples),
        "claim_reason": np.random.choice(
            ["Expired", "Damaged", "Wrong Item", "Missing Parts", "Not as Described"], n_samples
        ),
    }

    # --- Timestamps ---
    base_date = datetime.datetime(2024, 1, 1)
    order_ts = [
        base_date + datetime.timedelta(days=np.random.randint(0, 365))
        for _ in range(n_samples)
    ]
    data["order_timestamp"] = order_ts

    delivery_ts = [
        ts + datetime.timedelta(days=np.random.randint(1, 6)) for ts in order_ts
    ]
    data["delivery_timestamp"] = delivery_ts

    # Fraud flags
    fraud_flags = np.random.choice([0, 1], n_samples, p=[1 - fraud_ratio, fraud_ratio])

    # Claim timestamps (fraudulent claims tend to be suspiciously delayed)
    claim_ts = []
    for i in range(n_samples):
        if fraud_flags[i] == 0:  # Legitimate
            delay = np.random.randint(0, 31)
        else:  # Fraudulent
            delay = np.random.randint(25, 50)
        claim_ts.append(delivery_ts[i] + datetime.timedelta(days=delay))
    data["claim_timestamp"] = claim_ts
    data["fraud_flag"] = fraud_flags

    # --- Claim Descriptions (text for embeddings) ---
    legit_templates = [
        "The {product} I received was {issue}, please process a refund.",
        "I ordered a {product}, but it arrived {issue}.",
        "Refund requested: {product} came {issue} on arrival.",
        "The {product} does not work as expected because it's {issue}.",
        "Disappointed with the {product}, it was {issue}.",
    ]
    fraud_templates = [
        "I never got my {product}, refund immediately!",
        "Your {product} is {issue}, I demand money back.",
        "This is the third {issue} {product} I’ve got from you, refund now!",
        "How can you sell such a {issue} {product}? Full refund required.",
        "Worst {product} ever, totally {issue}, refund asap!",
    ]
    issues = ["damaged", "broken", "expired", "defective", "not as described"]

    descriptions = []
    for i in range(n_samples):
        product = data["product_category"][i].lower()
        issue = random.choice(issues)
        if fraud_flags[i] == 0:
            template = random.choice(legit_templates)
        else:
            template = random.choice(fraud_templates)
        descriptions.append(template.format(product=product, issue=issue))

    data["claim_description"] = descriptions

    # --- User history text (to teach the model)
    # Create histories with varying repetition of refunds and cues
    history_templates = [
        "no prior refunds",
        "requested refund last month for similar issue",
        "two refund requests in last 2 months",
        "multiple refund requests; again asking for refund",
        "refund refund refund in past history",
    ]

    user_histories = []
    for i in range(n_samples):
        # Base history by random choice
        base = random.choice(history_templates)
        # If fraudulent, increase repetition
        if fraud_flags[i] == 1:
            rep = random.randint(2, 6)
            hist = ("i want refund ") * rep + base
        else:
            hist = base
        user_histories.append(hist)
    data["user_history"] = user_histories

    df = pd.DataFrame(data)

    # Adjust fraud flag to correlate with history repetition (teach model the pattern)
    def history_signal(h: str) -> int:
        h = (h or "").lower()
        score = 0
        for kw in ["refund", "again", "multiple", "repeat"]:
            if kw in h:
                score += 1
        score += h.count("refund")
        return score

    hist_sig = df["user_history"].apply(history_signal)
    # Increase probability of fraud when history signal is high
    boost_mask = hist_sig >= 3
    df.loc[boost_mask, "fraud_flag"] = 1

    return df


if __name__ == "__main__":
    df = generate_synthetic_data(20, 0.2)
    print(df.head())
