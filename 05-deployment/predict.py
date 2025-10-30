# 10 different customer profiles for testing
sample_customers = [
    # 1. Young customer with monthly contract and high charges
    {
        'gender': 'female',
        'seniorcitizen': 0,
        'partner': 'no',
        'dependents': 'no',
        'phoneservice': 'yes',
        'multiplelines': 'yes',
        'internetservice': 'fiber_optic',
        'onlinesecurity': 'no',
        'onlinebackup': 'no',
        'deviceprotection': 'no',
        'techsupport': 'no',
        'streamingtv': 'yes',
        'streamingmovies': 'yes',
        'contract': 'month-to-month',
        'paperlessbilling': 'yes',
        'paymentmethod': 'electronic_check',
        'tenure': 1,
        'monthlycharges': 90.5,
        'totalcharges': 90.5
    },
    
    # 2. Long-term customer with full services
    {
        'gender': 'male',
        'seniorcitizen': 1,
        'partner': 'yes',
        'dependents': 'yes',
        'phoneservice': 'yes',
        'multiplelines': 'yes',
        'internetservice': 'fiber_optic',
        'onlinesecurity': 'yes',
        'onlinebackup': 'yes',
        'deviceprotection': 'yes',
        'techsupport': 'yes',
        'streamingtv': 'yes',
        'streamingmovies': 'yes',
        'contract': 'two_year',
        'paperlessbilling': 'yes',
        'paymentmethod': 'bank_transfer_(automatic)',
        'tenure': 72,
        'monthlycharges': 120.35,
        'totalcharges': 8665.2
    },
    
    # 3. Basic service customer
    {
        'gender': 'male',
        'seniorcitizen': 0,
        'partner': 'no',
        'dependents': 'no',
        'phoneservice': 'yes',
        'multiplelines': 'no',
        'internetservice': 'dsl',
        'onlinesecurity': 'yes',
        'onlinebackup': 'no',
        'deviceprotection': 'no',
        'techsupport': 'no',
        'streamingtv': 'no',
        'streamingmovies': 'no',
        'contract': 'month-to-month',
        'paperlessbilling': 'no',
        'paymentmethod': 'mailed_check',
        'tenure': 24,
        'monthlycharges': 45.3,
        'totalcharges': 1087.2
    },
    
    # 4. Senior citizen with premium services
    {
        'gender': 'female',
        'seniorcitizen': 1,
        'partner': 'yes',
        'dependents': 'no',
        'phoneservice': 'yes',
        'multiplelines': 'yes',
        'internetservice': 'fiber_optic',
        'onlinesecurity': 'yes',
        'onlinebackup': 'yes',
        'deviceprotection': 'yes',
        'techsupport': 'yes',
        'streamingtv': 'yes',
        'streamingmovies': 'yes',
        'contract': 'one_year',
        'paperlessbilling': 'yes',
        'paymentmethod': 'credit_card_(automatic)',
        'tenure': 48,
        'monthlycharges': 110.5,
        'totalcharges': 5304.0
    },
    
    # 5. No internet service customer
    {
        'gender': 'male',
        'seniorcitizen': 0,
        'partner': 'yes',
        'dependents': 'yes',
        'phoneservice': 'yes',
        'multiplelines': 'yes',
        'internetservice': 'no',
        'onlinesecurity': 'no_internet_service',
        'onlinebackup': 'no_internet_service',
        'deviceprotection': 'no_internet_service',
        'techsupport': 'no_internet_service',
        'streamingtv': 'no_internet_service',
        'streamingmovies': 'no_internet_service',
        'contract': 'two_year',
        'paperlessbilling': 'no',
        'paymentmethod': 'mailed_check',
        'tenure': 36,
        'monthlycharges': 25.75,
        'totalcharges': 927.0
    },
    
    # 6. New customer with all services
    {
        'gender': 'female',
        'seniorcitizen': 0,
        'partner': 'no',
        'dependents': 'no',
        'phoneservice': 'yes',
        'multiplelines': 'yes',
        'internetservice': 'fiber_optic',
        'onlinesecurity': 'yes',
        'onlinebackup': 'yes',
        'deviceprotection': 'yes',
        'techsupport': 'yes',
        'streamingtv': 'yes',
        'streamingmovies': 'yes',
        'contract': 'month-to-month',
        'paperlessbilling': 'yes',
        'paymentmethod': 'electronic_check',
        'tenure': 2,
        'monthlycharges': 115.8,
        'totalcharges': 231.6
    },
    
    # 7. Mid-term DSL customer
    {
        'gender': 'female',
        'seniorcitizen': 0,
        'partner': 'yes',
        'dependents': 'yes',
        'phoneservice': 'yes',
        'multiplelines': 'no',
        'internetservice': 'dsl',
        'onlinesecurity': 'yes',
        'onlinebackup': 'yes',
        'deviceprotection': 'no',
        'techsupport': 'yes',
        'streamingtv': 'no',
        'streamingmovies': 'no',
        'contract': 'one_year',
        'paperlessbilling': 'no',
        'paymentmethod': 'bank_transfer_(automatic)',
        'tenure': 12,
        'monthlycharges': 60.85,
        'totalcharges': 730.2
    },
    
    # 8. Phone-only senior customer
    {
        'gender': 'male',
        'seniorcitizen': 1,
        'partner': 'yes',
        'dependents': 'no',
        'phoneservice': 'yes',
        'multiplelines': 'no',
        'internetservice': 'no',
        'onlinesecurity': 'no_internet_service',
        'onlinebackup': 'no_internet_service',
        'deviceprotection': 'no_internet_service',
        'techsupport': 'no_internet_service',
        'streamingtv': 'no_internet_service',
        'streamingmovies': 'no_internet_service',
        'contract': 'two_year',
        'paperlessbilling': 'no',
        'paymentmethod': 'mailed_check',
        'tenure': 60,
        'monthlycharges': 20.65,
        'totalcharges': 1239.0
    },
    
    # 9. High-risk customer profile
    {
        'gender': 'female',
        'seniorcitizen': 0,
        'partner': 'no',
        'dependents': 'no',
        'phoneservice': 'yes',
        'multiplelines': 'yes',
        'internetservice': 'fiber_optic',
        'onlinesecurity': 'no',
        'onlinebackup': 'no',
        'deviceprotection': 'no',
        'techsupport': 'no',
        'streamingtv': 'yes',
        'streamingmovies': 'yes',
        'contract': 'month-to-month',
        'paperlessbilling': 'yes',
        'paymentmethod': 'electronic_check',
        'tenure': 3,
        'monthlycharges': 95.25,
        'totalcharges': 285.75
    },
    
    # 10. Low-risk customer profile
    {
        'gender': 'male',
        'seniorcitizen': 0,
        'partner': 'yes',
        'dependents': 'yes',
        'phoneservice': 'yes',
        'multiplelines': 'yes',
        'internetservice': 'fiber_optic',
        'onlinesecurity': 'yes',
        'onlinebackup': 'yes',
        'deviceprotection': 'yes',
        'techsupport': 'yes',
        'streamingtv': 'yes',
        'streamingmovies': 'yes',
        'contract': 'two_year',
        'paperlessbilling': 'no',
        'paymentmethod': 'credit_card_(automatic)',
        'tenure': 55,
        'monthlycharges': 115.35,
        'totalcharges': 6344.25
    }
]

# Load the model and make predictions
import pickle

with open('05-deployment/model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)

# Make predictions for all customers
for i, customer in enumerate(sample_customers, 1):
    churn_probability = model.predict_proba([customer])[0, 1]
    print(f'Customer {i} churn probability: {churn_probability:.3f}')