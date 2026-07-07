import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.vector_store import load_vector_store

vs = load_vector_store()
results = vs.similarity_search("ai là chủ sở hữu thửa đất 169", k=4)
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
