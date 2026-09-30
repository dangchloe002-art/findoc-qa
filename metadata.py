"""
Metadata enrichment for 10-K chunks.

Each chunk gets document-level fields (company, form type, fiscal year,
source file) and a section label inferred from the 10-K "Item" headings
(e.g. Item 1A -> Risk Factors, Item 8 -> Financial Statements). The
section label lets retrieval filter or boost by part of the filing.
"""

import re

# Standard Form 10-K item titles (Regulation S-K).
TENK_ITEMS = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Common Equity and Stockholder Matters",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis (MD&A)",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Beneficial Owners and Management",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

ITEM_HEADING = re.compile(r"^\s*Item\s+(\d{1,2}[A-C]?)\.", re.MULTILINE | re.IGNORECASE)
# After the numbered items, a 10-K has a signature page and then the
# attached exhibit documents (indentures, stock plans, ...).
SIGNATURES_HEADING = re.compile(r"^\s*SIGNATURES\s*$", re.MULTILINE)
EXHIBIT_HEADING = re.compile(r"^\s*Exhibit\s+\d+(\.\d+)?\s*$", re.MULTILINE)

FRONT_MATTER = "Front Matter"
SIGNATURES = "Signatures"
EXHIBIT_DOCS = "Attached Exhibit Documents"


def _items_in(text: str) -> list[str]:
    """Return the item codes of all 'Item N.' headings in a text, in order."""
    return [m.group(1).upper() for m in ITEM_HEADING.finditer(text)
            if m.group(1).upper() in TENK_ITEMS]


def find_toc_pages(chunks: list[dict], min_items: int = 5) -> set[int]:
    """
    Find table-of-contents pages.

    A TOC page lists many items at once, starting from Item 1. A real
    content page can also hold several short items (e.g. Part III items
    that are incorporated by reference), but those pages do not start
    again from Item 1.
    """
    items_by_page: dict[int, set[str]] = {}
    for c in chunks:
        items_by_page.setdefault(c["page_num"], set()).update(_items_in(c["text"]))
    return {p for p, items in items_by_page.items()
            if len(items) >= min_items and "1" in items}


def infer_sections(chunks: list[dict]) -> list[dict]:
    """
    Tag every chunk with 'section_item' and 'section' based on the most
    recent Item heading seen in reading order. Chunks before the first
    heading, and chunks on TOC pages, are labelled 'Front Matter'. The
    signature page and the exhibit documents attached after it get their
    own labels, so they do not inherit the last item (Item 16).
    Returns new dicts; the input list is not modified.
    """
    toc_pages = find_toc_pages(chunks)
    ordered = sorted(chunks, key=lambda c: (c["page_num"], c["chunk_index"]))

    current = None          # current item code, e.g. "1A"
    trailing = None         # SIGNATURES or EXHIBIT_DOCS once past the items
    tagged = {}
    for c in ordered:
        text = c["text"]
        if c["page_num"] in toc_pages:
            item, label = None, FRONT_MATTER
        else:
            if trailing is None:
                items = _items_in(text)
                if items:
                    # A chunk that contains a heading belongs to the last
                    # section that starts inside it.
                    current = items[-1]
                if current and SIGNATURES_HEADING.search(text):
                    trailing = SIGNATURES
            elif trailing == SIGNATURES and EXHIBIT_HEADING.search(text):
                trailing = EXHIBIT_DOCS

            if trailing:
                item, label = None, trailing
            elif current:
                item, label = current, TENK_ITEMS[current]
            else:
                item, label = None, FRONT_MATTER

        new = dict(c)
        new["section_item"] = item or ""
        new["section"] = label
        tagged[c["id"]] = new

    return [tagged[c["id"]] for c in chunks]


def enrich_chunks(chunks: list[dict], company: str, fiscal_year: int,
                  source_file: str, form_type: str = "10-K") -> list[dict]:
    """Add document-level fields and section labels to every chunk."""
    enriched = infer_sections(chunks)
    for c in enriched:
        c["company"] = company
        c["form_type"] = form_type
        c["fiscal_year"] = fiscal_year
        c["source_file"] = source_file
    return enriched
