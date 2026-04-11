"""
Dataset 5: Planogram JSON Generator
Creates realistic retail planogram data in industry-standard format.
Maps M5 dataset categories/items to shelf positions.
"""

import json
import os
import random

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "planograms")

# Product catalog based on M5 dataset departments
DEPARTMENTS = {
    "FOODS": {
        "FOODS_1": {  # Bakery/Deli
            "name": "Bakery & Deli",
            "products": [
                {"sku": "FOODS_1_001", "name": "White Bread Loaf", "price": 2.49, "width_cm": 12, "height_cm": 18, "depth_cm": 12, "facing": 2},
                {"sku": "FOODS_1_002", "name": "Whole Wheat Bread", "price": 3.29, "width_cm": 12, "height_cm": 18, "depth_cm": 12, "facing": 2},
                {"sku": "FOODS_1_003", "name": "Hamburger Buns 8pk", "price": 2.99, "width_cm": 20, "height_cm": 8, "depth_cm": 15, "facing": 3},
                {"sku": "FOODS_1_004", "name": "Hot Dog Buns 8pk", "price": 2.79, "width_cm": 20, "height_cm": 8, "depth_cm": 15, "facing": 2},
                {"sku": "FOODS_1_005", "name": "Tortillas 10pk", "price": 3.49, "width_cm": 22, "height_cm": 2, "depth_cm": 22, "facing": 4},
            ]
        },
        "FOODS_2": {  # Frozen/Refrigerated
            "name": "Frozen Foods",
            "products": [
                {"sku": "FOODS_2_001", "name": "Frozen Pizza Pepperoni", "price": 5.99, "width_cm": 28, "height_cm": 4, "depth_cm": 28, "facing": 3},
                {"sku": "FOODS_2_002", "name": "Frozen Vegetables Mix", "price": 2.49, "width_cm": 15, "height_cm": 20, "depth_cm": 8, "facing": 4},
                {"sku": "FOODS_2_003", "name": "Ice Cream Vanilla 1L", "price": 4.99, "width_cm": 12, "height_cm": 15, "depth_cm": 12, "facing": 3},
                {"sku": "FOODS_2_004", "name": "Frozen French Fries", "price": 3.49, "width_cm": 15, "height_cm": 25, "depth_cm": 10, "facing": 3},
                {"sku": "FOODS_2_005", "name": "Frozen Chicken Nuggets", "price": 6.99, "width_cm": 20, "height_cm": 25, "depth_cm": 10, "facing": 2},
            ]
        },
        "FOODS_3": {  # Canned/Packaged
            "name": "Canned & Packaged",
            "products": [
                {"sku": "FOODS_3_001", "name": "Canned Tomatoes 400g", "price": 1.29, "width_cm": 8, "height_cm": 12, "depth_cm": 8, "facing": 6},
                {"sku": "FOODS_3_002", "name": "Pasta Spaghetti 500g", "price": 1.49, "width_cm": 7, "height_cm": 28, "depth_cm": 5, "facing": 5},
                {"sku": "FOODS_3_003", "name": "Canned Tuna 170g", "price": 2.19, "width_cm": 9, "height_cm": 4, "depth_cm": 9, "facing": 8},
                {"sku": "FOODS_3_004", "name": "Rice Long Grain 1kg", "price": 3.29, "width_cm": 15, "height_cm": 22, "depth_cm": 8, "facing": 4},
                {"sku": "FOODS_3_005", "name": "Cereal Corn Flakes", "price": 4.49, "width_cm": 22, "height_cm": 30, "depth_cm": 8, "facing": 3},
            ]
        },
    },
    "HOUSEHOLD": {
        "HOUSEHOLD_1": {  # Cleaning
            "name": "Cleaning Supplies",
            "products": [
                {"sku": "HOUSEHOLD_1_001", "name": "All-Purpose Cleaner 750ml", "price": 3.99, "width_cm": 8, "height_cm": 28, "depth_cm": 6, "facing": 4},
                {"sku": "HOUSEHOLD_1_002", "name": "Dish Soap 500ml", "price": 2.49, "width_cm": 7, "height_cm": 22, "depth_cm": 5, "facing": 5},
                {"sku": "HOUSEHOLD_1_003", "name": "Laundry Detergent 2L", "price": 8.99, "width_cm": 15, "height_cm": 25, "depth_cm": 10, "facing": 3},
                {"sku": "HOUSEHOLD_1_004", "name": "Paper Towels 6-roll", "price": 6.49, "width_cm": 30, "height_cm": 25, "depth_cm": 15, "facing": 2},
                {"sku": "HOUSEHOLD_1_005", "name": "Trash Bags 30-count", "price": 5.99, "width_cm": 20, "height_cm": 30, "depth_cm": 8, "facing": 3},
            ]
        },
        "HOUSEHOLD_2": {  # Personal Care
            "name": "Personal Care",
            "products": [
                {"sku": "HOUSEHOLD_2_001", "name": "Shampoo 400ml", "price": 5.49, "width_cm": 7, "height_cm": 22, "depth_cm": 5, "facing": 4},
                {"sku": "HOUSEHOLD_2_002", "name": "Body Wash 500ml", "price": 4.99, "width_cm": 8, "height_cm": 20, "depth_cm": 6, "facing": 4},
                {"sku": "HOUSEHOLD_2_003", "name": "Toothpaste 150g", "price": 3.49, "width_cm": 5, "height_cm": 18, "depth_cm": 4, "facing": 6},
                {"sku": "HOUSEHOLD_2_004", "name": "Deodorant Spray", "price": 4.29, "width_cm": 5, "height_cm": 16, "depth_cm": 5, "facing": 5},
                {"sku": "HOUSEHOLD_2_005", "name": "Hand Soap 250ml", "price": 2.99, "width_cm": 7, "height_cm": 15, "depth_cm": 5, "facing": 5},
            ]
        },
    },
    "HOBBIES": {
        "HOBBIES_1": {  # Arts & Crafts
            "name": "Arts & Crafts",
            "products": [
                {"sku": "HOBBIES_1_001", "name": "Colored Pencils 24pk", "price": 8.49, "width_cm": 20, "height_cm": 12, "depth_cm": 2, "facing": 4},
                {"sku": "HOBBIES_1_002", "name": "Sketch Pad A4", "price": 5.99, "width_cm": 22, "height_cm": 30, "depth_cm": 1, "facing": 5},
                {"sku": "HOBBIES_1_003", "name": "Acrylic Paint Set", "price": 12.99, "width_cm": 25, "height_cm": 15, "depth_cm": 4, "facing": 2},
                {"sku": "HOBBIES_1_004", "name": "Glue Sticks 4pk", "price": 3.49, "width_cm": 10, "height_cm": 15, "depth_cm": 3, "facing": 6},
                {"sku": "HOBBIES_1_005", "name": "Craft Scissors", "price": 4.29, "width_cm": 8, "height_cm": 22, "depth_cm": 1, "facing": 5},
            ]
        },
        "HOBBIES_2": {  # Games/Toys
            "name": "Games & Toys",
            "products": [
                {"sku": "HOBBIES_2_001", "name": "Playing Cards Deck", "price": 3.99, "width_cm": 7, "height_cm": 12, "depth_cm": 2, "facing": 8},
                {"sku": "HOBBIES_2_002", "name": "Puzzle 500pc", "price": 9.99, "width_cm": 25, "height_cm": 20, "depth_cm": 5, "facing": 3},
                {"sku": "HOBBIES_2_003", "name": "Board Game Classic", "price": 14.99, "width_cm": 30, "height_cm": 25, "depth_cm": 6, "facing": 2},
                {"sku": "HOBBIES_2_004", "name": "Toy Car Set", "price": 7.99, "width_cm": 20, "height_cm": 15, "depth_cm": 5, "facing": 3},
                {"sku": "HOBBIES_2_005", "name": "Coloring Book Kids", "price": 4.49, "width_cm": 22, "height_cm": 28, "depth_cm": 1, "facing": 5},
            ]
        },
    },
}

# Store configurations
STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]


def create_shelf(shelf_id: int, dept_key: str, dept_info: dict, num_shelves: int = 4) -> dict:
    """Create a shelf unit with multiple shelf levels."""
    shelf_width_cm = 120  # Standard gondola width
    shelf_height_cm = 200
    shelf_depth_cm = 45

    shelves = []
    products = dept_info["products"]

    for level in range(num_shelves):
        level_products = []
        x_pos = 0

        # Distribute products across each shelf level
        for prod in products:
            for _ in range(prod["facing"]):
                level_products.append({
                    "sku": prod["sku"],
                    "name": prod["name"],
                    "price": prod["price"],
                    "position": {
                        "x": round(x_pos, 1),
                        "y": round(level * (shelf_height_cm / num_shelves), 1),
                        "width": prod["width_cm"],
                        "height": prod["height_cm"],
                        "depth": prod["depth_cm"],
                    },
                    "expected_quantity": random.randint(3, 12),
                    "min_quantity": 2,
                    "reorder_point": 3,
                })
                x_pos += prod["width_cm"]
                if x_pos >= shelf_width_cm:
                    break
            if x_pos >= shelf_width_cm:
                break

        shelves.append({
            "level": level,
            "height_from_ground_cm": round(level * (shelf_height_cm / num_shelves), 1),
            "products": level_products,
        })

    return {
        "shelf_id": f"SHELF_{shelf_id:03d}",
        "department": dept_key,
        "department_name": dept_info["name"],
        "dimensions": {
            "width_cm": shelf_width_cm,
            "height_cm": shelf_height_cm,
            "depth_cm": shelf_depth_cm,
        },
        "num_levels": num_shelves,
        "shelves": shelves,
    }


def create_store_planogram(store_id: str) -> dict:
    """Create a complete planogram for a store."""
    shelf_counter = 0
    aisles = []

    for cat_name, departments in DEPARTMENTS.items():
        aisle_shelves = []
        for dept_key, dept_info in departments.items():
            shelf_counter += 1
            shelf = create_shelf(shelf_counter, dept_key, dept_info)
            aisle_shelves.append(shelf)

        aisles.append({
            "aisle_id": f"AISLE_{cat_name}",
            "category": cat_name,
            "shelves": aisle_shelves,
        })

    return {
        "store_id": store_id,
        "planogram_version": "1.0",
        "generated_at": "2026-04-11T00:00:00Z",
        "total_shelves": shelf_counter,
        "total_skus": sum(
            len(dept["products"])
            for departments in DEPARTMENTS.values()
            for dept in departments.values()
        ),
        "aisles": aisles,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate planogram for each store
    all_planograms = {}
    for store_id in STORES:
        random.seed(hash(store_id))  # Reproducible per-store variation
        planogram = create_store_planogram(store_id)
        all_planograms[store_id] = planogram

        # Save individual store planogram
        store_path = os.path.join(OUTPUT_DIR, f"planogram_{store_id}.json")
        with open(store_path, "w") as f:
            json.dump(planogram, f, indent=2)
        print(f"  [OK] {store_path} ({planogram['total_shelves']} shelves, {planogram['total_skus']} SKUs)")

    # Save combined planogram index
    index = {
        "stores": list(all_planograms.keys()),
        "total_stores": len(all_planograms),
        "departments": {
            cat: list(depts.keys())
            for cat, depts in DEPARTMENTS.items()
        },
        "product_catalog": {
            sku: {
                "name": prod["name"],
                "price": prod["price"],
                "department": dept_key,
                "category": cat_name,
            }
            for cat_name, departments in DEPARTMENTS.items()
            for dept_key, dept_info in departments.items()
            for prod in dept_info["products"]
            for sku in [prod["sku"]]
        },
    }

    index_path = os.path.join(OUTPUT_DIR, "planogram_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\n  [OK] Index saved: {index_path}")
    print(f"     {index['total_stores']} stores, {len(index['product_catalog'])} unique SKUs")


if __name__ == "__main__":
    print("=" * 50)
    print("ShelfMind AI - Planogram Generator")
    print("=" * 50)
    main()
