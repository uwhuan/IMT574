import pandas as pd
import numpy as np
import itertools

df = pd.read_csv('./data/stayzilla_com-travel_sample.csv')
# Only select columns that are related to our project
df = df[['additional_info', 'amenities', 'city', 'description', 'image_count', 'latitude', 
    'longitude', 'occupancy', 'property_id', 'property_name', 'property_type', 'room_price', 
    'room_types', 'service_value']]

# Clean up room_price, occupancy, additional_info and amenities column
df = df.assign(
    room_price=df.room_price.map(
        lambda v: float(v[:v.find("per")]) if not pd.isnull(v) else np.nan
    ),
    acceptance_rate=df.additional_info.map(
        lambda v: v.split("|")[0].split(":")[-1].split(" ")[0] if not pd.isnull(v) else np.nan
    ),
    response_time=df.additional_info.map(
        lambda v: v.split("|")[-1].split(":")[-1] if not pd.isnull(v) else np.nan
    ),
    adult_occupancy=df.occupancy.map(
        lambda v: float(v.split(" ")[0]) if not pd.isnull(v) else np.nan
    ),
    child_occupancy=df.occupancy.map(
        lambda v: float(v.split(" ")[-2]) if not pd.isnull(v) else np.nan
    ),
    service_value=df.service_value.map(
        lambda v: np.nan if pd.isnull(v) else v if v in ['Not Verified', 'Verified'] else np.nan
    ),
    amenity_split=df.amenities.fillna("").map(
        lambda f: [am.strip() for am in f.split("|")]
    )
)

# Split the items in amanities into separate columns
top_amenities = pd.Series(
    list(itertools.chain(*(df.amenities.fillna("").map(lambda f: [am.strip() for am in f.split("|")])
                            .values.tolist())))).value_counts().head(13).index.values
top_amenities = [am for am in top_amenities if am != ""] 

for amenity in top_amenities:
    df[amenity] = df.amenity_split.map(lambda l: amenity in l)

# We can see that the Car Parking is duplicated with Parking
df['Parking'] = (df['Car Parking'] | df['Parking'])

# Drop unnecessary columns
df.drop(['amenity_split', 'amenities', 'additional_info', 'occupancy', 'Car Parking'], axis=1, inplace=True)

# Save to csv file
df.to_csv('./data/stayzilla_rough_cleaned.csv', index=False)