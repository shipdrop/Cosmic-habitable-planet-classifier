import json
import data_loader

points = data_loader.get_sample_planets(limit=100)

with open("archive-data.json") as f:
    json.dump({"planets": points}, f, indent=2)

print("Saved archive-data.json successfully!")