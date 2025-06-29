import pandas as pd

def preprocess(df, max_len=500):
    if "Name" in df.columns:
        df = df.rename(columns={"Name": "name"})
    if "Sequence full length" in df.columns:
        df = df.rename(columns={"Sequence full length": "sequence"})
    if "Label" in df.columns:
        df = df.rename(columns={"Label": "label"})
    
    all_names = df["name"].values.tolist()
    all_names = [n.strip() for n in all_names]
    all_names = [n[n.find("(")+1:n.rfind(")")] if n.find("(")+1 != n.rfind(")") else " " for n in all_names]
    all_names = [n[:n.find("/")] if n.find("/") != -1 else n for n in all_names]
    all_names = [n.lower() for n in all_names]
    all_names = pd.Series(all_names).value_counts()
    all_names = all_names[all_names >= 8]
    df['label'] = df['name'].apply(lambda x: x[x.find("(")+1:x.find(")")] if x.find("(")+1 != x.find(")") else " ")
    df['label'] = df['label'].apply(lambda x: x[:x.find("/")] if x.find("/") != -1 else x)
    df['label'] = df['label'].apply(lambda x: x.lower())
    df = df[df['label'] != " "]
    df = df[df['label'].isin(all_names.index)]
    # reset index
    df = df.reset_index(drop=True)
    # if the seq len > 1000, drop the row
    df = df[df["sequence"].apply(lambda x: len(x)) < max_len]
    print(list(set(df['label'])))
    return df

def train_eval_split(df, val_ratio, random_state):
    train_df = []
    val_df = []
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        val_label_data = label_df.sample(frac=val_ratio, random_state=random_state)
        train_label_data = label_df.drop(val_label_data.index)
        train_df.append(train_label_data)
        val_df.append(val_label_data)
    train_df = pd.concat(train_df)
    val_df = pd.concat(val_df)
    return train_df, val_df
