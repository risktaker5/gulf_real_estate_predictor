import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime

# Custom country-city mappings
COUNTRY_CITIES = {
    'AE': ['Dubai', 'Abu Dhabi', 'Sharjah', 'Ajman', 'Ras Al Khaimah'],
    'SA': ['Riyadh', 'Jeddah', 'Mecca', 'Medina', 'Dammam'],
    'QA': ['Doha', 'Al Wakrah', 'Al Khor', 'Umm Salal', 'Al Rayyan'],
    'BH': ['Manama', 'Muharraq', 'Riffa', 'Hamad Town', 'Aali'],
    'OM': ['Muscat', 'Salalah', 'Sohar', 'Nizwa', 'Sur'],
    'IN': ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai'],
    'US': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
}

def generate_real_estate_dataset(num_samples=1000):
    fake = Faker()
    
    # Generate countries with appropriate distribution
    countries = np.random.choice(
        ['AE', 'SA', 'QA', 'BH', 'OM', 'IN', 'US'],
        size=num_samples,
        p=[0.25, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1]  # Higher probability for Gulf countries
    )
    
    # Generate corresponding cities
    cities = [np.random.choice(COUNTRY_CITIES[country]) for country in countries]
    
    # Base features
    data = {
        'country': countries,
        'city': cities,
        'sqft': np.random.randint(800, 10000, size=num_samples),
        'bedrooms': np.random.choice([1, 2, 3, 4, 5, 6], size=num_samples, p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05]),
        'bathrooms': np.round(np.random.uniform(1, 4, size=num_samples), 1),
        'year_built': np.random.randint(1990, 2023, size=num_samples),
        'lot_size': np.random.randint(1000, 20000, size=num_samples),
        'walk_score': np.random.randint(0, 100, size=num_samples),
        'crime_rate': np.round(np.random.uniform(0.1, 10.0, size=num_samples), 1),
        'school_rating': np.random.randint(1, 10, size=num_samples),
        'has_pool': np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3]),
        'has_gym': np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5]),
        'has_maid_room': np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4]),
        'near_mosque': np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6]),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Country-specific adjustments
    # Higher prices for UAE/Qatar, maid rooms more common in Gulf
    df.loc[df['country'].isin(['AE', 'QA']), 'has_maid_room'] = np.random.choice([0, 1], size=len(df[df['country'].isin(['AE', 'QA'])]), p=[0.3, 0.7])
    
    # Calculate derived features
    current_year = datetime.now().year
    df['age'] = current_year - df['year_built']
    
    # Generate synthetic price with country-based variations
    base_price = df['sqft'] * np.random.uniform(100, 1000, size=num_samples)
    
    # Apply country multipliers
    country_multipliers = {
        'AE': 1.5,  # UAE most expensive
        'QA': 1.4,   # Qatar
        'SA': 1.2,   # Saudi
        'BH': 1.1,   # Bahrain
        'OM': 1.0,   # Oman
        'IN': 0.4,   # India
        'US': 1.0    # US baseline
    }
    
    df['price'] = base_price * df['country'].map(country_multipliers)
    
    # Apply city premium (e.g., Dubai more expensive than other UAE cities)
    city_premiums = {
        'Dubai': 1.5, 'Abu Dhabi': 1.3, 'Riyadh': 1.4, 'Doha': 1.6,
        'Manama': 1.1, 'Muscat': 1.0, 'Mumbai': 1.2, 'New York': 1.8
    }
    
    for city, premium in city_premiums.items():
        df.loc[df['city'] == city, 'price'] *= premium
    
    # Final price adjustments based on features
    df['price'] = df['price'] * (1 + df['bedrooms'] * 0.1) * \
                  (1 + df['bathrooms'] * 0.05) * \
                  (1 + df['school_rating'] * 0.05) * \
                  (1 - df['crime_rate'] * 0.01) * \
                  (1 + df['has_pool'] * 0.15 + df['has_gym'] * 0.1)
    
    # Calculate derived price features
    df['price_per_sqft'] = (df['price'] / df['sqft']).round(2)
    df['price_per_bedroom'] = (df['price'] / df['bedrooms']).round(2)
    df['bath_bed_ratio'] = (df['bathrooms'] / df['bedrooms']).round(2)
    
    # Reorder columns to match model expectations
    expected_columns = [
        'sqft', 'bedrooms', 'bathrooms', 'year_built', 'lot_size',
        'walk_score', 'crime_rate', 'school_rating', 'has_pool',
        'has_gym', 'has_maid_room', 'near_mosque', 'country', 'city',
        'price_per_sqft', 'age', 'price_per_bedroom', 'bath_bed_ratio',
        'price'
    ]
    
    return df[expected_columns]

# Generate and save dataset
if __name__ == "__main__":
    dataset = generate_real_estate_dataset(5000)
    dataset.to_csv('gulf_real_estate_data.csv', index=False)
    print("Dataset generated with Gulf countries focus")