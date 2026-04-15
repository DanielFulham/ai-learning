def print_query_results(results, fields=None):
    for i, (doc_id, document, distance) in enumerate(zip(
        results['ids'][0], results['documents'][0], results['distances'][0]
    )):
        metadata = results['metadatas'][0][i]
        print(f"  {i+1}. {metadata['title']} ({doc_id}) - Distance: {float(distance):.4f}")
        if fields:
            field_output = ", ".join(f"{f}: {metadata.get(f, 'N/A')}" for f in fields)
            print(f"     {field_output}")
        print(f"     Document: {document[:100]}...")


def print_get_results(results, fields):
    for i, doc_id in enumerate(results['ids']):
        metadata = results['metadatas'][i]
        field_output = ", ".join(f"{f}: {metadata.get(f, 'N/A')}" for f in fields)
        print(f"  {i+1}. {metadata['title']} ({field_output}) - ID: {doc_id}")