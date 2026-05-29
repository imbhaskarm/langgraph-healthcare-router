from typing import Literal
from pydantic import BaseModel, Field


class QueryCategory(BaseModel):
    """Structured output schema for department classification."""
    category: Literal["billing", "appointments", "records", "insurance"] = Field(
        description="The healthcare support category of the user query"
    )


class QuerySentiment(BaseModel):
    """Structured output schema for sentiment classification."""
    sentiment: Literal["positive", "neutral", "negative", "distress"] = Field(
        description="The sentiment of the user query"
    )
