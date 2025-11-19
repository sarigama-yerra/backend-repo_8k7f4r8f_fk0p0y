import os
import hashlib
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from apscheduler.schedulers.background import BackgroundScheduler

from database import db, create_document, get_documents

# -------------------- FastAPI app --------------------
app = FastAPI(title="Job Scraper Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Models --------------------
class CreateUser(BaseModel):
    email: EmailStr
    keywords: List[str] = []
    notify_webhook: Optional[str] = None

class UserOut(BaseModel):
    email: EmailStr
    keywords: List[str]
    notify_webhook: Optional[str] = None
    is_active: bool = True

class JobOut(BaseModel):
    source: str
    title: str
    url: str
    company: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    posted_at: Optional[datetime] = None
    keywords: List[str] = []

# -------------------- Utils --------------------
USER_AGENT = {
    "User-Agent": "Mozilla/5.0 (compatible; JobScraperBot/1.0; +https://example.com/bot)"
}

def make_fingerprint(source: str, url: str, title: str) -> str:
    base = f"{source}|{url}|{title}".encode("utf-8")
    return hashlib.sha256(base).hexdigest()


def save_job_if_new(job: Dict[str, Any]) -> Optional[str]:
    """Save job if fingerprint not yet in DB. Returns inserted id or None if exists."""
    fp = job.get("fingerprint")
    if not fp:
        job["fingerprint"] = make_fingerprint(job.get("source", ""), job.get("url", ""), job.get("title", ""))
        fp = job["fingerprint"]
    # check existence
    exists = get_documents("job", {"fingerprint": fp}, limit=1)
    if exists:
        return None
    return create_document("job", job)


def contains_any(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)


# -------------------- Scrapers --------------------

def scrape_reddit(keyword: str) -> List[Dict[str, Any]]:
    url = f"https://www.reddit.com/search.json?q={requests.utils.quote(keyword + ' hiring job')}&limit=20&sort=new"
    try:
        r = requests.get(url, headers=USER_AGENT, timeout=15)
        r.raise_for_status()
        data = r.json()
        jobs = []
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            title = d.get("title", "")
            post_url = f"https://www.reddit.com{d.get('permalink', '')}"
            created = d.get("created_utc")
            posted_at = datetime.fromtimestamp(created, tz=timezone.utc) if created else None
            if not contains_any(title, [keyword]):
                continue
            job = {
                "source": "reddit",
                "title": title,
                "url": post_url,
                "company": None,
                "location": None,
                "description": d.get("selftext"),
                "posted_at": posted_at,
                "keywords": [keyword],
            }
            job["fingerprint"] = make_fingerprint(job["source"], job["url"], job["title"])
            jobs.append(job)
        return jobs
    except Exception:
        return []


def scrape_myjobmag(keyword: str) -> List[Dict[str, Any]]:
    search_url = f"https://www.myjobmag.com/search/jobs?search={requests.utils.quote(keyword)}"
    jobs: List[Dict[str, Any]] = []
    try:
        resp = requests.get(search_url, headers=USER_AGENT, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.select("a.job-listing-title, a.job-title, a[href*='/job/']"):
            title = a.get_text(strip=True)
            href = a.get("href")
            if not href:
                continue
            url = href if href.startswith("http") else f"https://www.myjobmag.com{href}"
            if not contains_any(title, [keyword]):
                continue
            job = {
                "source": "myjobmag",
                "title": title,
                "url": url,
                "company": None,
                "location": None,
                "description": None,
                "posted_at": None,
                "keywords": [keyword],
            }
            job["fingerprint"] = make_fingerprint(job["source"], job["url"], job["title"])
            jobs.append(job)
        return jobs
    except Exception:
        return []


def scrape_linkedin(keyword: str) -> List[Dict[str, Any]]:
    """LinkedIn scraping is restricted; return empty to avoid ToS violations."""
    return []


def scrape_upwork(keyword: str) -> List[Dict[str, Any]]:
    """Upwork feed often requires authenticated RSS; return empty for safety."""
    return []


def aggregate_scrape(keywords: List[str]) -> List[Dict[str, Any]]:
    all_jobs: List[Dict[str, Any]] = []
    for kw in keywords:
        all_jobs.extend(scrape_reddit(kw))
        all_jobs.extend(scrape_myjobmag(kw))
        # placeholders for completeness without violating ToS
        all_jobs.extend(scrape_linkedin(kw))
        all_jobs.extend(scrape_upwork(kw))
    return all_jobs


# -------------------- Notifications --------------------
import smtplib
from email.mime.text import MIMEText

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USER or "no-reply@example.com")


def send_email(to_email: str, subject: str, body: str) -> bool:
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
        # Fallback: store notifications collection
        create_document("notification", {
            "email": to_email,
            "subject": subject,
            "body": body,
            "sent": False,
            "reason": "SMTP not configured",
        })
        return False
    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = FROM_EMAIL
        msg["To"] = to_email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(FROM_EMAIL, [to_email], msg.as_string())
        create_document("notification", {
            "email": to_email,
            "subject": subject,
            "body": body,
            "sent": True,
        })
        return True
    except Exception as e:
        create_document("notification", {
            "email": to_email,
            "subject": subject,
            "body": body,
            "sent": False,
            "reason": str(e)[:200],
        })
        return False


# -------------------- Scheduler job --------------------

def run_job_scrape_and_notify():
    # 1) Load users
    users = get_documents("user")
    if not users:
        return {"status": "no-users"}

    # 2) Compile unique keywords
    keywords: List[str] = []
    for u in users:
        if u.get("is_active", True):
            for k in u.get("keywords", []):
                if k and k not in keywords:
                    keywords.append(k)

    if not keywords:
        return {"status": "no-keywords"}

    # 3) Scrape
    jobs = aggregate_scrape(keywords)
    new_jobs: List[Dict[str, Any]] = []
    for j in jobs:
        inserted = save_job_if_new(j)
        if inserted:
            new_jobs.append(j)

    # 4) Notify users
    if new_jobs:
        # build per-user matches
        for u in users:
            if not u.get("is_active", True):
                continue
            email = u.get("email")
            ukeywords = [k.lower() for k in u.get("keywords", [])]
            matches = [j for j in new_jobs if contains_any(j.get("title", ""), ukeywords)]
            if not matches:
                continue
            lines = [f"- [{j['source']}] {j['title']}\n  {j['url']}" for j in matches[:15]]
            body = "New jobs matching your keywords:\n\n" + "\n".join(lines)
            send_email(email, "New job opportunities found", body)

    return {"status": "ok", "new_jobs": len(new_jobs)}


scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(run_job_scrape_and_notify, "interval", minutes=10, id="job-scraper", replace_existing=True)
scheduler.start()


# -------------------- API Routes --------------------
@app.get("/")
def root():
    return {"message": "Job Scraper API running", "scheduler": "every 10 minutes"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


@app.post("/api/users", response_model=UserOut)
def create_user(user: CreateUser):
    # dedupe by email
    existing = get_documents("user", {"email": user.email}, limit=1)
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    doc = {
        "email": user.email,
        "keywords": [k.strip() for k in user.keywords if k.strip()],
        "notify_webhook": user.notify_webhook,
        "is_active": True,
    }
    create_document("user", doc)
    return doc


@app.get("/api/users", response_model=List[UserOut])
def list_users():
    users = get_documents("user")
    # pydantic validation
    return [
        {
            "email": u.get("email"),
            "keywords": u.get("keywords", []),
            "notify_webhook": u.get("notify_webhook"),
            "is_active": u.get("is_active", True),
        }
        for u in users
    ]


@app.post("/api/trigger")
def trigger_scrape():
    res = run_job_scrape_and_notify()
    return res


@app.get("/api/jobs", response_model=List[JobOut])
def get_jobs(keyword: Optional[str] = None, source: Optional[str] = None, limit: int = Query(50, ge=1, le=200)):
    filt: Dict[str, Any] = {}
    if source:
        filt["source"] = source
    docs = get_documents("job", filt, limit=limit)
    out: List[Dict[str, Any]] = []
    for d in docs:
        if keyword and not contains_any((d.get("title") or "") + " " + (d.get("description") or ""), [keyword]):
            continue
        out.append({
            "source": d.get("source"),
            "title": d.get("title"),
            "url": d.get("url"),
            "company": d.get("company"),
            "location": d.get("location"),
            "description": d.get("description"),
            "posted_at": d.get("posted_at"),
            "keywords": d.get("keywords", []),
        })
    return out


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
