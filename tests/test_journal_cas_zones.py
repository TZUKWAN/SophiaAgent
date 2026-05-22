import pytest
from sophia.research.journal_db import JournalDatabase

@pytest.fixture
def db():
    return JournalDatabase()

def test_find_by_issn(db):
    result = db.find_by_issn("0028-0836")
    assert result is None  # Because we just test the new method, currently journal_db doesn't have 0028-0836 in its journals.json

def test_find_by_name_exact(db):
    result = db.find_by_name("Non-existent Journal", fuzzy=False)
    assert len(result) == 0

def test_find_by_name_fuzzy(db):
    result = db.find_by_name("Non-existent", fuzzy=True)
    assert len(result) == 0

def test_get_cas_zone_by_name_exact(db):
    result = db.get_cas_zone("Nature")
    assert result is not None
    assert result["zone"] == 1
    assert result["issn"] == "0028-0836"
    assert result["category"] == "综合"

def test_get_cas_zone_by_name_case_insensitive(db):
    result = db.get_cas_zone("nature")
    assert result is not None
    assert result["zone"] == 1
    assert result["top_journal"] is True

def test_get_cas_zone_by_issn(db):
    result = db.get_cas_zone("0162-8828")
    assert result is not None
    assert result["zone"] == 1
    assert result["category"] == "计算机科学"

def test_get_cas_zone_not_found(db):
    result = db.get_cas_zone("unknown-journal-name")
    assert result is None
