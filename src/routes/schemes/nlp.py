from pydantic import BaseModel
from typing import Optional

class PushRequest(BaseModel):
    do_reset: Optional[int] = 0

class SearchRequest(BaseModel):
    text: str
    limit: Optional[int] = 20
    similarity_threshold: Optional[float] = None  # <--- threshold in [0..1]
    use_rerank: Optional[bool] = False           # <--- optional re-rank flag
