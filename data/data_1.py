import requests
import pandas as pd

# Replace with your RapidAPI credentials
RAPIDAPI_KEY = "14624c46e1mshf7c3d0e2a8c6620p103816jsn0cc49565e415"
RAPIDAPI_HOST = "bayut.p.rapidapi.com"

# API Endpoint for listing properties for sale
url = "https://bayut.p.rapidapi.com/properties/list"

querystring = {
    "locationExternalIDs": "5002",  # UAE
    "purpose": "for-sale",
    "hitsPerPage": "50",
    "page": "0",
    "lang": "en"
}

headers = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST
}

all_listings = []

# Loop through multiple pages
for page in range(0, 5):  # Fetch 5 pages (250 listings)
    print(f"Fetching page {page + 1}...")
    querystring["page"] = str(page)
    
    response = requests.get(url, headers=headers, params=querystring)
    
    if response.status_code == 200:
        data = response.json()
        hits = data.get("hits", [])
        for listing in hits:
            all_listings.append({
                "id": listing.get("id"),
                "title": listing.get("title"),
                "price": listing.get("price"),
                "bedrooms": listing.get("bedrooms"),
                "bathrooms": listing.get("bathrooms"),
                "area": listing.get("area"),
                "location": ', '.join([loc.get("name", "") for loc in listing.get("location", [])]),
                "agency": listing.get("agency", {}).get("name"),
                "furnishingStatus": listing.get("furnishingStatus"),
                "propertyType": listing.get("product", "")
            })
    else:
        print(f"Error {response.status_code}: {response.text}")
        break

# Save to CSV
df = pd.DataFrame(all_listings)
df.to_csv("uae_real_estate_data.csv", index=False)
print("✅ Data saved to uae_real_estate_data.csv")
