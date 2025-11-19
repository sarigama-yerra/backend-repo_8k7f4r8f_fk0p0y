"""
Database Schemas for Job Scraper

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Job -> "job" collection
- Keyword -> "keyword" collection
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime

class User(BaseModel):
    email: EmailStr = Field(..., description="User email to notify")
    keywords: List[str] = Field(default_factory=list, description="Keywords the user wants to track")
    notify_webhook: Optional[str] = Field(None, description="Optional webhook (e.g., Slack) to send notifications")
    is_active: bool = Field(True)

class Job(BaseModel):
    source: str = Field(..., description="Source name: upwork, linkedin, reddit, myjobmag, etc")
    title: str
    url: str
    company: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    posted_at: Optional[datetime] = None
    # normalized fields for dedup
    fingerprint: str = Field(..., description="Unique hash per job")
    keywords: List[str] = Field(default_factory=list)

class Keyword(BaseModel):
    term: str
    created_by: Optional[str] = Field(None, description="Email of the user who added it")
