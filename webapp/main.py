"""
Backend FastAPI cho giao diện web: 2 chức năng chính
1. Trích xuất: upload ảnh Sổ đỏ -> chạy pipeline OCR + rule-based extraction ->
   trả JSON + báo cáo Markdown dễ đọc (KHÔNG cần hỏi-đáp, trả ngay toàn bộ).
2. Hỏi-đáp: giữ lại chatbot cũ (Query Router -> JSON Lookup -> RAG + LLM) trên
   các tài liệu đã trích xuất, chạy qua Ollama LOCAL (server từ xa đang lỗi).

Chạy: .venv/Scripts/python.exe -m uvicorn webapp.main:app --reload --port 8000
"""
import json
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

# Pipeline in Windows print tiếng Việt ra stdout (VD "Đang xử lý..."); nếu
# console mặc định không phải UTF-8 (uvicorn không tự ép như các script CLI
# khác), print() sẽ crash UnicodeEncodeError giữa lúc xử lý request.
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import yaml
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

from src.pipeline import DocumentPipeline
from src.report_generator import generate_readable_report
from src.text_formatter import format_as_markdown
from src.query_router import QueryRouter
from src.synonym_expander import expand_query

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_DIR = os.path.join(BASE_DIR, "outputs", "predictions")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
MARKDOWNS_DIR = os.path.join(BASE_DIR, "outputs", "markdowns")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
# Ảnh của từng tài liệu (bản gốc upload + ảnh từng trang đã convert) - lưu bền để
# giao diện hiển thị và "xem lại ảnh" sau này. Cấu trúc:
#   outputs/uploads/<DOC_ID>/original/<file gốc>
#   outputs/uploads/<DOC_ID>/pages/page_01.png ...
UPLOADS_DIR = os.path.join(BASE_DIR, "outputs", "uploads")
CORPUS_DIR = os.path.join(BASE_DIR, "data", "documents")
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = FastAPI(title="Sổ đỏ OCR + Chat API")


def _list_document_images(doc_id):
    """URL ảnh từng trang của 1 tài liệu: ưu tiên ảnh đã upload qua web
    (outputs/uploads), fallback sang corpus gốc (data/documents) cho DOC_001-004."""
    pages_dir = os.path.join(UPLOADS_DIR, doc_id, "pages")
    if os.path.isdir(pages_dir):
        return [
            f"/uploads/{doc_id}/pages/{f}"
            for f in sorted(os.listdir(pages_dir))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    corpus_dir = os.path.join(CORPUS_DIR, doc_id)
    if os.path.isdir(corpus_dir):
        return [
            f"/corpus/{doc_id}/{f}"
            for f in sorted(os.listdir(corpus_dir))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    return []


def _is_extraction_empty(doc_json):
    """Phát hiện trường hợp OCR không đọc được chữ nào (ảnh trắng/mờ/lỗi/chọn
    nhầm file) - mọi field đều None và không có extra_fields/change_history -
    để cảnh báo rõ ràng thay vì trả về JSON rỗng im lặng."""
    holder = doc_json.get("holder") or {}
    land = doc_json.get("land_parcel") or {}
    asset = doc_json.get("asset") or {}
    all_none = (
        not any(holder.values())
        and not any(land.values())
        and not any(asset.values())
        and not doc_json.get("holders")
        and not doc_json.get("extra_fields")
        and not doc_json.get("change_history")
    )
    return all_none

_pipeline = None
_router = None
_vector_store = None
_llm_chain = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with open(os.path.join(BASE_DIR, "configs", "pipeline.yaml"), "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        _pipeline = DocumentPipeline(cfg)
    return _pipeline


def get_router():
    global _router
    if _router is None:
        _router = QueryRouter(predictions_dir=PREDICTIONS_DIR)
    else:
        # Reload để thấy tài liệu vừa trích xuất thêm (không cần restart server)
        _router.documents = _router._load_all_predictions()
    return _router


def get_llm_chain():
    """
    Trả về (chain, vector_store). `chain` (prompt+LLM) được cache vĩnh viễn -
    không phụ thuộc dữ liệu. `vector_store` được cache RIÊNG và có thể bị buộc
    tải lại (invalidate_vector_store()) sau khi có tài liệu mới nhúng, mà
    không cần khởi tạo lại LLM.
    """
    global _vector_store, _llm_chain
    if _llm_chain is None:
        from langchain_community.llms import Ollama
        from langchain_core.prompts import PromptTemplate

        llm = Ollama(base_url="https://ocr.devforenv.com", model="qwen3:8b", temperature=0.0)
        prompt = PromptTemplate.from_template(
            "Bạn là một trợ lý thông minh chuyên giải đáp các câu hỏi về thông tin Giấy chứng nhận quyền sử "
            "dụng đất (Sổ đỏ/Sổ hồng).\nDựa vào NỘI DUNG TRÍCH XUẤT dưới đây, hãy trả lời câu hỏi bằng tiếng "
            "Việt ngắn gọn, chính xác. Nếu không có thông tin, trả lời trung thực \"Tôi không tìm thấy thông "
            "tin này trong tài liệu hiện tại.\"\n\nNỘI DUNG TRÍCH XUẤT:\n{context}\n\nCÂU HỎI: {question}\n"
            "TRẢ LỜI:"
        )
        _llm_chain = prompt | llm

    if _vector_store is None:
        from src.vector_store import load_vector_store

        _vector_store = load_vector_store()

    return _llm_chain, _vector_store


def invalidate_vector_store():
    """Buộc load_vector_store() chạy lại ở lần chat kế tiếp, sau khi ChromaDB
    vừa được nhúng lại (xem rebuild_embeddings trong /api/extract)."""
    global _vector_store
    _vector_store = None


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
if os.path.isdir(CORPUS_DIR):
    app.mount("/corpus", StaticFiles(directory=CORPUS_DIR), name="corpus")


@app.on_event("startup")
def _warmup():
    """Nạp sẵn pipeline OCR (PaddleOCR + VietOCR) ngay khi server khởi động,
    thay vì để request /api/extract ĐẦU TIÊN gánh 8-10s tải model - giúp lần
    trích xuất đầu mượt như các lần sau. Chạy trong thread nền để không chặn
    server sẵn sàng nhận request."""
    import threading

    def _load():
        try:
            get_pipeline()
            print("[warmup] Pipeline OCR đã sẵn sàng.")
        except Exception as e:
            print(f"[warmup] Lỗi nạp pipeline: {e}")

    threading.Thread(target=_load, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/documents")
def list_documents():
    router = get_router()
    docs = []
    for doc_id, data in router.documents.items():
        holders = data.get("holders") or [data.get("holder", {})]
        names = [h.get("name") for h in holders if h.get("name")]
        docs.append({"doc_id": doc_id, "holders": names})
    docs.sort(key=lambda d: d["doc_id"])
    return {"documents": docs}


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    json_path = os.path.join(PREDICTIONS_DIR, f"{doc_id}.json")
    report_path = os.path.join(REPORTS_DIR, f"{doc_id}.md")
    if not os.path.exists(json_path):
        return JSONResponse({"error": f"Không tìm thấy tài liệu {doc_id}"}, status_code=404)
    with open(json_path, "r", encoding="utf-8") as f:
        doc_json = json.load(f)
    report = ""
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report = f.read()
    return {
        "doc_id": doc_id,
        "json": doc_json,
        "report_markdown": report,
        "images": _list_document_images(doc_id),
    }


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """
    Xoá VĨNH VIỄN một tài liệu: không chỉ ẩn khỏi web mà xoá thật toàn bộ dữ
    liệu trong các thư mục source - JSON trích xuất, báo cáo, markdown RAG,
    chunks, ảnh đã upload, ảnh gốc trong corpus (nếu có), và vectors trong
    ChromaDB. Frontend BẮT BUỘC hiện hộp thoại xác nhận trước khi gọi.
    """
    # Chống path traversal: doc_id đi thẳng vào đường dẫn file
    if not re.fullmatch(r"[A-Za-z0-9_-]+", doc_id):
        return JSONResponse({"error": "Mã tài liệu không hợp lệ."}, status_code=400)

    if not os.path.exists(os.path.join(PREDICTIONS_DIR, f"{doc_id}.json")):
        return JSONResponse({"error": f"Không tìm thấy tài liệu {doc_id}."}, status_code=404)

    deleted = []

    # 1. Các file kết quả
    for path, label in [
        (os.path.join(PREDICTIONS_DIR, f"{doc_id}.json"), "JSON trích xuất"),
        (os.path.join(REPORTS_DIR, f"{doc_id}.md"), "báo cáo"),
        (os.path.join(MARKDOWNS_DIR, f"{doc_id}.md"), "markdown RAG"),
        (os.path.join(BASE_DIR, "outputs", "chunks", f"{doc_id}_chunks.json"), "chunks"),
    ]:
        if os.path.exists(path):
            os.remove(path)
            deleted.append(label)

    # 2. Ảnh đã upload (bản gốc + từng trang)
    doc_upload_dir = os.path.join(UPLOADS_DIR, doc_id)
    if os.path.isdir(doc_upload_dir):
        shutil.rmtree(doc_upload_dir)
        deleted.append("ảnh đã upload")

    # 3. Ảnh gốc trong corpus data/documents (nếu tài liệu thuộc corpus)
    corpus_doc_dir = os.path.join(CORPUS_DIR, doc_id)
    if os.path.isdir(corpus_doc_dir):
        shutil.rmtree(corpus_doc_dir)
        deleted.append("ảnh gốc trong data/documents")

    # 4. Vectors trong ChromaDB - dùng chromadb client trực tiếp (không cần
    # nạp model embedding ~13s như load_vector_store, vì xoá không cần embed)
    chroma_dir = os.path.join(BASE_DIR, "outputs", "chroma_db")
    if os.path.isdir(chroma_dir):
        try:
            import chromadb

            client = chromadb.PersistentClient(path=chroma_dir)
            collection = client.get_collection("langchain")
            existing = collection.get(where={"document_id": doc_id})
            n_vec = len(existing.get("ids", []))
            if n_vec:
                collection.delete(where={"document_id": doc_id})
                deleted.append(f"{n_vec} vectors ChromaDB")
        except Exception as e:
            print(f"[delete] Bỏ qua ChromaDB ({e})")

        # Dọn parent_store.json (nội dung chunk cha lưu kèm DB)
        parent_file = os.path.join(chroma_dir, "parent_store.json")
        if os.path.exists(parent_file):
            try:
                with open(parent_file, "r", encoding="utf-8") as f:
                    parent_map = json.load(f)
                prefix = f"{doc_id}_"
                cleaned = {k: v for k, v in parent_map.items() if not k.startswith(prefix)}
                if len(cleaned) != len(parent_map):
                    with open(parent_file, "w", encoding="utf-8") as f:
                        json.dump(cleaned, f, ensure_ascii=False)
            except Exception:
                pass

    # Buộc vector store cache nạp lại ở lần chat kế tiếp
    invalidate_vector_store()

    return {"doc_id": doc_id, "deleted": deleted}


@app.post("/api/extract")
async def extract(
    files: List[UploadFile] = File(...),
    doc_id: Optional[str] = Form(None),
    embed: bool = Form(False),
):
    """
    Nhận 1 hoặc nhiều file (ảnh/PDF/Word, nhiều trang của cùng 1 sổ), tự động
    render PDF/Word thành ảnh + xoay các trang nằm ngang về khổ dọc, rồi chạy
    pipeline ngay - trả về JSON + Markdown dễ đọc, KHÔNG cần hỏi từng câu.

    embed=False (mặc định): CHỈ trích xuất (OCR + JSON + report), bỏ qua bước
    sinh Markdown RAG + nhúng ChromaDB - nhanh hơn đáng kể, phù hợp khi chỉ cần
    trích xuất thông tin, không cần hỏi-đáp về tài liệu này. Truyền embed=true
    nếu muốn hỏi-đáp được về tài liệu này ngay sau khi trích xuất.
    """
    tmp_dir = tempfile.mkdtemp(prefix="extract_")
    try:
        uploaded_paths = []
        for f in files:
            dest = os.path.join(tmp_dir, f.filename)
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            uploaded_paths.append(dest)
        uploaded_paths.sort()

        # doc_id dùng làm tên thư mục + URL ảnh -> sanitize về [A-Za-z0-9_-]
        # (tránh ký tự tiếng Việt/khoảng trắng gây lỗi đường dẫn/URL).
        raw_id = doc_id or os.path.splitext(os.path.basename(uploaded_paths[0]))[0]
        final_doc_id = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_id.upper()).strip("_") or "TAI_LIEU"

        from src.file_converter import convert_to_images

        saved_paths = []
        rotated_pages = []
        for uploaded in uploaded_paths:
            pages, rotated = convert_to_images(uploaded, tmp_dir)
            saved_paths.extend(pages)
            rotated_pages.extend(rotated)
        saved_paths.sort()

        # Lưu BỀN bản gốc + ảnh từng trang vào thư mục riêng của tài liệu - để
        # giao diện hiển thị ảnh ngay khi trích xuất và "xem lại ảnh" về sau
        # (đồng thời phục vụ chẩn đoán khi kết quả sai/trống).
        doc_upload_dir = os.path.join(UPLOADS_DIR, final_doc_id)
        if os.path.isdir(doc_upload_dir):
            shutil.rmtree(doc_upload_dir)
        orig_dir = os.path.join(doc_upload_dir, "original")
        pages_dir = os.path.join(doc_upload_dir, "pages")
        os.makedirs(orig_dir)
        os.makedirs(pages_dir)
        for src in uploaded_paths:
            shutil.copyfile(src, os.path.join(orig_dir, os.path.basename(src)))
        image_urls = []
        for i, page in enumerate(saved_paths, start=1):
            ext = os.path.splitext(page)[1].lower() or ".png"
            page_name = f"page_{i:02d}{ext}"
            shutil.copyfile(page, os.path.join(pages_dir, page_name))
            image_urls.append(f"/uploads/{final_doc_id}/pages/{page_name}")

        pipeline = get_pipeline()
        doc_json, page_blocks_dict = pipeline.process_document(final_doc_id, saved_paths)
        report_text = generate_readable_report(doc_json)

        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        os.makedirs(MARKDOWNS_DIR, exist_ok=True)

        with open(os.path.join(PREDICTIONS_DIR, f"{final_doc_id}.json"), "w", encoding="utf-8") as out:
            json.dump(doc_json, out, ensure_ascii=False, indent=2)
        with open(os.path.join(REPORTS_DIR, f"{final_doc_id}.md"), "w", encoding="utf-8") as out:
            out.write(report_text)

        note = "Đã lưu JSON + báo cáo."
        if rotated_pages:
            note += f" Đã tự động xoay {len(rotated_pages)} trang nằm ngang: {', '.join(rotated_pages)}."

        if _is_extraction_empty(doc_json):
            note = (
                "⚠️ CẢNH BÁO: OCR không đọc được chữ nào từ ảnh vừa đưa vào (kết quả trống hoàn toàn). "
                "Kiểm tra lại ảnh có bị trắng/mờ/lỗi, chọn nhầm file, hoặc bị xoay sai chiều không, rồi thử lại. " + note
            )

        if embed:
            # Sinh Markdown RAG + nhúng THÊM TỪNG PHẦN (chỉ đúng tài liệu này,
            # không đụng tới các tài liệu khác đã có) - CHỈ làm khi được yêu
            # cầu rõ (embed=True), vì bước này tốn thêm thời gian (chunking +
            # model embedding BGE-M3) mà chỉ cần thiết cho tab Hỏi-đáp.
            md_text = format_as_markdown(page_blocks_dict, document_id=final_doc_id, doc_json=doc_json)
            with open(os.path.join(MARKDOWNS_DIR, f"{final_doc_id}.md"), "w", encoding="utf-8") as out:
                out.write(md_text)

            from src.embedding_pipeline import add_document_embedding

            note += " Đã nhúng vào ChromaDB - có thể hỏi-đáp ngay về tài liệu này."
            try:
                add_document_embedding(
                    final_doc_id,
                    markdowns_dir=MARKDOWNS_DIR, chunks_dir=os.path.join(BASE_DIR, "outputs", "chunks"),
                    chroma_dir=os.path.join(BASE_DIR, "outputs", "chroma_db"),
                )
                invalidate_vector_store()
            except Exception as e:
                note = f"Đã lưu JSON/báo cáo, nhưng nhúng vào ChromaDB thất bại ({e}) - hỏi-đáp có thể chưa thấy tài liệu này."
        else:
            note += " (Chưa nhúng vào ChromaDB - tick \"Nhúng để hỏi-đáp\" nếu cần hỏi-đáp về tài liệu này.)"

        return {
            "doc_id": final_doc_id,
            "json": doc_json,
            "report_markdown": report_text,
            "note": note,
            "images": image_urls,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


class ChatRequest(BaseModel):
    question: str
    reset_memory: bool = False


@app.post("/api/chat")
def chat(req: ChatRequest):
    router = get_router()
    if req.reset_memory:
        router.last_doc_id = None
        router.last_person_name = None

    agg_answer, _ = router.lookup_aggregate(req.question)
    if agg_answer:
        return {"method": "Aggregate", "answer": agg_answer}

    route_type, field_name = router.classify(req.question)
    if route_type == "field":
        answer, _ = router.lookup_json(req.question, field_name)
        if answer:
            return {"method": f"JSON Lookup ({field_name})", "answer": answer}

    extra_answer, _ = router.lookup_extra_field(req.question)
    if extra_answer:
        return {"method": "Extra Field Lookup", "answer": extra_answer}

    try:
        chain, vector_store = get_llm_chain()
    except Exception as e:
        return {"method": "error", "answer": f"Không kết nối được LLM/Vector DB: {e}"}

    expanded_query = expand_query(req.question)
    if router.last_doc_id:
        expanded_query = f"[Đang hỏi về tài liệu {router.last_doc_id}] {expanded_query}"
    results = vector_store.similarity_search(expanded_query, k=4)
    context_chunks = [
        f"[Nguồn: {d.metadata.get('document_id', 'Không rõ')}]:\n{d.page_content}" for d in results
    ]
    combined_context = "\n\n---\n\n".join(dict.fromkeys(context_chunks))

    try:
        response = chain.invoke({"context": combined_context, "question": req.question})
        answer = router.grounding_check(response.strip())
        return {"method": "RAG + LLM", "answer": answer}
    except Exception as e:
        return {"method": "error", "answer": f"Lỗi gọi LLM: {e}"}
