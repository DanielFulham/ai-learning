"""
Shamrock Rovers FC — Fan Segmentation Demo
==========================================
Proof of concept for the IT volunteer call.

Shows how vector similarity search + metadata filtering
could work on top of Supporter 360 fan data.

Data here is fictional but mirrors the shape of what
Future Ticketing + GoCardless + Shopify would produce.

"""

import chromadb
from chromadb.utils import embedding_functions
from rovers_fans import fans

# --- Setup ---

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.Client()
collection_name = "rovers_fans"


def main():
    try:
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Shamrock Rovers supporter profiles"},
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": ef,
            }
        )

        print("=== Shamrock Rovers Fan Segmentation Demo ===\n")
        print("Building fan profile collection...")

        # Serialise each fan into a natural language profile for embedding.
        # This is what makes semantic search work — structured data becomes
        # a sentence the embedding model can reason about.
        fan_documents = []
        for fan in fans:
            doc = f"{fan['membership_type'].replace('_', ' ').title()} supporter"
            doc += f" who attended {fan['matches_attended_2024']} matches in 2024"
            doc += f" and {fan['matches_attended_2023']} matches in 2023."
            doc += f" Has been a member for {fan['seasons_as_member']} seasons."
            doc += f" Opponents attended: {fan['opponents_attended']}."
            doc += f" Merchandise purchases: {fan['merchandise_purchases']}."
            if fan['derby_matches_only']:
                doc += " Attends Dublin derby matches only."
            if fan['lapsed']:
                doc += " Previously attended regularly but has not attended this season."
            fan_documents.append(doc)

        # Add fans to the collection.
        # Documents (natural language) drive semantic search.
        # Metadatas (raw fields) drive hard filters.
        collection.add(
            ids=[fan["id"] for fan in fans],
            documents=fan_documents,
            metadatas=[{
                "name": fan["name"],
                "membership_type": fan["membership_type"],
                "season_ticket_holder": fan["season_ticket_holder"],
                "seasons_as_member": fan["seasons_as_member"],
                "matches_2024": fan["matches_attended_2024"],
                "matches_2023": fan["matches_attended_2023"],
                "derby_only": fan["derby_matches_only"],
                "merchandise_purchases": fan["merchandise_purchases"],
                "location": fan["location"],
                "lapsed": fan["lapsed"],
            } for fan in fans]
        )

        print(f"Loaded {len(fans)} fan profiles into collection.\n")

        run_demo_queries(collection)

    except Exception as error:
        print(f"Error: {error}")


def run_demo_queries(collection):
    """
    Three demo queries for the IT call:

    1. Semantic search — find fans who behave like season ticket buyers
    2. Metadata filter  — find derby-only fans (the specific question asked)
    3. Combined         — find high-attendance casual fans to target for ST conversion
    """

    # ------------------------------------------------------------------
    # DEMO 1: Semantic similarity — "who looks like a season ticket buyer?"
    # ------------------------------------------------------------------
    print("--- Demo 1: Semantic Search ---")
    print("Question: Which fans have a profile most similar to a loyal season ticket holder?\n")

    query = "loyal supporter who attends most home matches every season, long-term member, buys merchandise"
    results = collection.query(
        query_texts=[query],
        n_results=4
    )

    for i, (doc_id, distance) in enumerate(zip(
        results['ids'][0], results['distances'][0]
    )):
        metadata = results['metadatas'][0][i]
        print(f"  {i+1}. {metadata['name']} ({metadata['location']})")
        print(f"     Membership: {metadata['membership_type']} | "
              f"2024 attendance: {metadata['matches_2024']} | "
              f"Similarity: {1 - distance:.2%}")

    # ------------------------------------------------------------------
    # DEMO 2: Metadata filter — derby-only fans
    # This is the "fans who only attended Dublin derbies" question.
    # Pure filter — no semantic search needed here.
    # ------------------------------------------------------------------
    print("\n--- Demo 2: Metadata Filter ---")
    print("Question: Which fans only attend Dublin derby matches?\n")

    results = collection.get(
        where={"derby_only": True}
    )

    print(f"  Found {len(results['ids'])} derby-only fans:")
    for i, fan_id in enumerate(results['ids']):
        metadata = results['metadatas'][i]
        total = metadata['matches_2024'] + metadata['matches_2023']
        print(f"  - {metadata['name']} ({metadata['location']}) — "
              f"{total} derby matches across 2023/24")

    print("\n  → These fans would respond to 'guaranteed derby seat' messaging with a season ticket.")

    # ------------------------------------------------------------------
    # DEMO 3: Combined — high-potential casual fans for ST conversion
    # Semantic: profile looks like someone close to committing
    # Filter:   not already a ST holder, attended 6+ matches in 2024
    # ------------------------------------------------------------------
    print("\n--- Demo 3: Combined Search (Semantic + Filter) ---")
    print("Question: Which casual fans attended enough matches in 2024 that")
    print("          a season ticket would have saved them money?\n")

    query = "fan who attends multiple matches per season and watches a variety of opponents"
    results = collection.query(
        query_texts=[query],
        n_results=5,
        where={
            "$and": [
                {"season_ticket_holder": False},
                {"lapsed": False},
                {"matches_2024": {"$gte": 6}}
            ]
        }
    )

    print(f"  Found {len(results['ids'][0])} fans:")
    for i, (fan_id, distance) in enumerate(zip(
        results['ids'][0], results['distances'][0]
    )):
        metadata = results['metadatas'][0][i]
        print(f"  {i+1}. {metadata['name']} ({metadata['location']})")
        print(f"     {metadata['matches_2024']} matches in 2024 | "
              f"Merch purchases: {metadata['merchandise_purchases']} | "
              f"Match: {1 - distance:.2%}")

    print("\n  → These are your highest-priority season ticket conversion targets.")

    # ------------------------------------------------------------------
    # DEMO 4: Lapsed fans — re-engagement targets
    # ------------------------------------------------------------------
    print("\n--- Demo 4: Lapsed Fan Re-engagement ---")
    print("Question: Which fans attended regularly before but haven't come back this season?\n")

    results = collection.get(
        where={
            "$and": [
                {"lapsed": True},
                {"matches_2023": {"$gte": 8}}
            ]
        }
    )

    print(f"  Found {len(results['ids'])} lapsed fans worth re-engaging:")
    for i, fan_id in enumerate(results['ids']):
        metadata = results['metadatas'][i]
        print(f"  - {metadata['name']} ({metadata['location']}) — "
              f"{metadata['matches_2023']} matches in 2023, 0 in 2024")

    print("\n  → These fans knew the club well. A targeted 'we miss you' offer")
    print("    is more effective than a generic campaign.")

    print("\n=== End of Demo ===")
    print("\nNote: This runs on 15 fictional fans.")
    print("The same code connects to Supporter 360's PostgreSQL database")
    print("and scales to the full supporter base without changes.")


if __name__ == "__main__":
    main()