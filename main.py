import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from setting import *

# Establish PostgreSQL database connection
db_connection = psycopg2.connect(
    host = HOST,      # PostgreSQL server address (localhost or IP)
    user = USER,          # PostgreSQL username (admin as per your earlier command)
    password = PASS,      # PostgreSQL password (admin as per your earlier command)
    port = PORT,             # Default PostgreSQL port
    database = DB       # Specify your PostgreSQL database name here
)

# Modify SQL query to join users and discount_usage_history
users_query = """
SELECT u.user_id, u.name, u.email, u.total_spent, u.registration_date, uh.discount_id
FROM users u
LEFT JOIN discount_usage_history uh ON u.user_id = uh.user_id;
"""

# Load data into DataFrame
users = pd.read_sql(users_query, db_connection)

# Check if data was successfully loaded
print("Users DataFrame:")
print(users.head())
print(f"Users shape: {users.shape}")

# Drop rows where discount_id is NaN (user hasn't used any discount)
users = users.dropna(subset=['discount_id'])

# If there are no users with discount_id, print a message and exit
if users.shape[0] == 0:
    print("No users with discount_id found. Exiting...")
    exit()

# Data processing
# If total_spent is greater than 10000, the user is likely to receive more discounts
users['total_spent'] = users['total_spent'].apply(lambda x: 1 if x > 10000 else 0)

# Convert registration_date to days since a reference date (1970-01-01)
users['registration_date'] = pd.to_datetime(users['registration_date'], errors='coerce')
users['registration_date'] = (users['registration_date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')

# Features (X) and Target (y)
X = users[['total_spent', 'registration_date']]  # Input features
y = users['discount_id']  # Target: discount codes used

# Check the shapes of X and y
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the split data
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Predict discount code for a new user
new_user = pd.DataFrame({'total_spent': [1], 'registration_date': [20230101]})
predicted_discount = model.predict(new_user)
print("Predicted Discount Code ID:", predicted_discount)
