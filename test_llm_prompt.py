import os
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

llm = Ollama(model="qwen2.5:1.5b", temperature=0.0)

markdown_context = """
Cá nhân hoặc tổ chức có tên ghi tại mục I là chủ sở hữu nhà ở và sử dụng đất ở
Phạm Văn Thông | Mục I - Chủ sở hữu nhà ở và sử dụng đất ở
Bà: Hồ Lệ Hồng | - Sinh năm: 1959 | CMND số: 020168965
"""

prompt_template = """Bạn là trợ lý thông minh. Dựa vào nội dung dưới đây, trả lời câu hỏi:
{context}

CÂU HỎI: {question}

LƯU Ý QUAN TRỌNG:
- Người ký giấy (VD: Phạm Văn Thông) KHÔNG PHẢI chủ sở hữu.
- Chủ sở hữu thường có chữ "Bà:" hoặc "Ông:" đứng trước tên.

TRẢ LỜI:"""

prompt = PromptTemplate.from_template(prompt_template)
chain = prompt | llm

response = chain.invoke({"context": markdown_context, "question": "ai là chủ sở hữu"})
print(response)
