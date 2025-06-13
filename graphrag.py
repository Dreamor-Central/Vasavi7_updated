import re
import asyncio
import logging
from neo4j import AsyncGraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEO4J_URI="neo4j+s://49757aa1.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="nx4Fv03abe3MPtjtIz8Nr-sm3jaXihT6dz2meq-DgxU"

driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Known product categories or product keywords in your catalog
KNOWN_CATEGORIES = [
    'tshirt', 'jacket', 'shirt', 'jeans', 'dress', 'shoes', 'coat', 'hat', 'bag', 'trousers', 'shorts'
]

async def extract_filters(query: str):
    """
    Extract category/product and max_price from a free-form natural language query.
    Returns a dict of filters.
    """
    filters = {}

    query_lower = query.lower()

    # Extract max price
    price_patterns = [
        r'budget\s*(?:is|:)?\s*₹?(\d+)',
        r'price\s*(?:is|:)?\s*₹?(\d+)',
        r'under\s*₹?(\d+)',
        r'below\s*₹?(\d+)',
        r'less than\s*₹?(\d+)',
        r'max(?:imum)?\s*₹?(\d+)',
        r'₹(\d+)',
        r'rs\.?\s*(\d+)'
    ]

    max_price = None
    for pattern in price_patterns:
        match = re.search(pattern, query_lower)
        if match:
            max_price = float(match.group(1))
            break
    if max_price:
        filters['max_price'] = max_price

    # Detect category/product keyword anywhere in the query
    found_category = None
    for cat in KNOWN_CATEGORIES:
        if re.search(r'\b' + re.escape(cat) + r's?\b', query_lower):
            found_category = cat
            break
    if found_category:
        filters['category'] = found_category

    return filters

def generate_cypher(filters: dict):
    """
    Generate Cypher query with optional filters.
    """
    base_match = """
    MATCH (p:Product)-[:BELONGS_TO]->(c:Category),
          (p)-[:HAS_FABRIC]->(f:Fabric)
    """

    where_clauses = []
    params = {}

    if 'category' in filters:
        where_clauses.append(
            "replace(toLower(c.name), '-', '') CONTAINS replace(toLower($category), '-', '')"
        )
        params['category'] = filters['category']

    if 'max_price' in filters:
        where_clauses.append("p.price < $max_price")
        params['max_price'] = filters['max_price']

    where_clause = ""
    if where_clauses:
        where_clause = "WHERE " + " AND ".join(where_clauses)

    cypher = f"""
    {base_match}
    {where_clause}
    RETURN p.name AS product, p.description AS description, p.price AS price, p.link AS link, c.name AS category, f.description AS fabric
    ORDER BY p.price ASC
    LIMIT 10
    """

    return cypher.strip(), params

async def query_neo4j(cypher: str, params: dict):
    async with driver.session() as session:
        result = await session.run(cypher, params)
        records = await result.data()
        return records

def format_response(records):
    if not records:
        return "I'm not sure based on the current data."

    lines = []
    for rec in records:
        lines.append(f"**{rec['product']} (₹{rec['price']})**")
        lines.append(f"- Category: {rec['category']}, Fabric: {rec['fabric']}")
        lines.append(f"- {rec['description']}")
        lines.append(f"- [Product Link]({rec['link']})")
        lines.append("")
    return "\n".join(lines)

async def graphrag_response(user_query: str) -> str:
    logger.info(f"User query: {user_query}")
    filters = await extract_filters(user_query)
    logger.info(f"Extracted filters: {filters}")

    if not filters:
        return ("I'm not sure based on the current data.")

    cypher, params = generate_cypher(filters)
    logger.info(f"Generated Cypher:\n{cypher}")

    records = await query_neo4j(cypher, params)
    response = format_response(records)
    return response

async def main():
    print("Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("Enter your query: ").strip()
        if user_input.lower() in ('exit', 'quit'):
            break
        response = await graphrag_response(user_input)
        print("\nResponse:\n" + response + "\n")

if __name__ == "__main__":
    asyncio.run(main())