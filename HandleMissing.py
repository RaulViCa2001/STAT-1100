
import altair as alt
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split



# Simplify working with large datasets in Altair
alt.data_transformers.disable_max_rows();
churn = pd.read_csv("data/Telco.csv");
#print((churn == ' ').sum())
churn['TotalCharges'] = pd.to_numeric(churn['TotalCharges'], errors='coerce')

# Check which rows are NaN now
blank_total = churn[churn['TotalCharges'].isnull()]
#print(blank_total)
#print(churn.loc[churn['tenure']==0, ['customerID', 'tenure', 'TotalCharges']])
churn = churn.dropna(subset=['TotalCharges'])
churn_model = churn.drop(columns=['customerID'])

# 4. List categorical columns to encode
categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

# 5. Apply one-hot encoding (drop_first avoids redundant columns)
churn_encoded = pd.get_dummies(churn_model, columns=categorical_cols, drop_first=True)

# 6. Encode target column 'Churn' as 0/1
churn_encoded['Churn'] = churn_encoded['Churn'].map({'Yes':1, 'No':0})
#print(churn_encoded.head())
#print(churn_encoded.info())
churn_yes = churn_encoded[churn_encoded['Churn'] == 1]
churn_no = churn_encoded[churn_encoded['Churn'] == 0]

# Oversample minority
churn_yes_upsampled = resample(churn_yes,
                               replace=True,
                               n_samples=len(churn_no),
                               random_state=42)

# Combine back
churn_balanced = pd.concat([churn_no, churn_yes_upsampled])

# Shuffle
churn_balanced = churn_balanced.sample(frac=1, random_state=42)
churn_encoded['AvgChargesPerMonth'] = churn_encoded['TotalCharges'] / (churn_encoded['tenure'] + 1)
churn_encoded['TenureBucket'] = pd.cut(churn_encoded['tenure'],
                                       bins=[0, 12, 24, 48, 72],
                                       labels=[1, 2, 3, 4])
# Correlation check
#print(churn_encoded[['AvgChargesPerMonth', 'TenureBucket', 'Churn']].corr())
X = churn_encoded.drop(columns=['Churn'])
y = churn_encoded['Churn']

# First, split train vs temp (85% train+validation, 15% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Then, split train vs validation (70/15/15 total)
# 70/85 ≈ 0.8235 for train from train_val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
)

# Check churn distribution
print("Train churn %:", y_train.mean())
print("Validation churn %:", y_val.mean())
print("Test churn %:", y_test.mean())