import pandas as pd
from neo4j import GraphDatabase
import os

# --- Neo4j Configuration ---
NEO4J_URI="neo4j+s://49757aa1.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="nx4Fv03abe3MPtjtIz8Nr-sm3jaXihT6dz2meq-DgxU"

# --- File Path Configuration ---
csv_path = os.path.join(os.path.dirname(__file__), 'vasavi3.csv')

# --- Load and Validate Data ---
try:
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_columns = ['Style Name', 'Description', 'Price', 'Product Link', 'Category', 'Fabric Description']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing required columns in CSV")
    
    # Validate Price column
    if not df['Price'].apply(lambda x: isinstance(x, (int, float)) and x >= 0).all():
        raise ValueError("Invalid Price values in CSV")
    
except FileNotFoundError:
    print(f" Error: CSV file not found at {csv_path}")
    exit(1)
except Exception as e:
    print(f" Error loading CSV: {str(e)}")
    exit(1)

# --- Connect to Neo4j ---
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
except Exception as e:
    print(f"Error connecting to Neo4j: {str(e)}")
    exit(1)

# --- Create Graph ---
def create_product_graph(tx, rows):
    """Create product graph nodes and relationships for a batch of rows."""
    for row in rows:
        tx.run("""
            MERGE (product:Product {name: $style_name})
            SET product.description = $description,
                product.price = $price,
                product.link = $link
            MERGE (category:Category {name: $category})
            MERGE (fabric:Fabric {description: $fabric_description})
            MERGE (product)-[:BELONGS_TO]->(category)
            MERGE (product)-[:HAS_FABRIC]->(fabric)
        """, 
        style_name=row['Style Name'],
        description=row['Description'],
        price=float(row['Price']),
        link=row['Product Link'],
        category=row['Category'],
        fabric_description=row['Fabric Description'])

# --- Insert into Neo4j ---
try:
    with driver.session() as session:
        batch_size = 10
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size].to_dict('records')
            session.execute_write(create_product_graph, batch)
    print("✅ Knowledge graph successfully created in Neo4j.")
except Exception as e:
    print(f" Error inserting into Neo4j: {str(e)}")
finally:
    driver.close()

