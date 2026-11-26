from dataclasses import dataclass, field


@dataclass
class Vacancy:
    refnr: str
    title: str
    company: str
    location: str
    url: str
    source: str
    published: str = ""
    description: str = ""
    score: float | None = None
    translation: str = ""
    score_state: str = "unscored"
    last_seen_at: str = ""
    description_fetched_at: str = ""
    description_hash: str = ""
    source_status: str = "active"
    archived_at: str = ""
    source_metadata: dict = field(default_factory=dict)
