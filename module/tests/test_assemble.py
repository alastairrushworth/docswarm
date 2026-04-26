from datetime import date

from pdf_to_json.assemble import assemble, empty_document
from pdf_to_json.schema import Document


def test_empty_document_validates():
    doc = empty_document()
    assert isinstance(doc, Document)
    assert doc.articles == []


def test_assemble_basic():
    meta = {
        "editor": "L. J. Berger",
        "issue": {"date": "1892-06-03", "volume": 5, "number": 18},
        "publisher": {"name": "N. H. Van Sicklen", "address": "57 Plymouth Place"},
        "cost": {"annual": "$2.00", "semiannual": "$1.00"},
    }
    articles = [
        {"title": "That's So!", "text": ["paragraph"], "pages": [1], "kind": "prose"},
        {"title": "Both Were Pleased.", "text": ["line one", "line two"], "pages": [1], "kind": "verse"},
    ]
    doc = assemble(meta, articles)
    assert doc.magazine.editor == "L. J. Berger"
    assert doc.magazine.issue.date == date(1892, 6, 3)
    assert len(doc.articles) == 2
    assert doc.articles[1].kind == "verse"


def test_drops_titleless_articles():
    doc = assemble({}, [{"title": "", "text": ["x"], "pages": [1]}])
    assert doc.articles == []
