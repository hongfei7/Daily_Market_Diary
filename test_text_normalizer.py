import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "market_diary"))

from modules.text_normalizer import normalize_news_text


def main() -> None:
    assert normalize_news_text("Abu Dhabi\u00e2\u20ac\u2122s Axight Adds to Gulf M&amp;A Wave", strip_html_tags=True) == "Abu Dhabi's Axight Adds to Gulf M&A Wave"
    assert normalize_news_text("<p>Hong Kong&nbsp;market&nbsp;update</p>", strip_html_tags=True) == "Hong Kong market update"
    assert normalize_news_text("Microsoft\u00e2\u20ac\u2122s stock\u00e2\u20ac\u201cwith an extreme bounce", strip_html_tags=True) == "Microsoft's stock-with an extreme bounce"
    assert normalize_news_text("Some\u00e8\u0081\u00bdheadline", strip_html_tags=True) == "Some headline"
    assert normalize_news_text("Uber Raises Delivery Hero Stake in \u00e2\u201a\u00ac70 Million Prosus Deal", strip_html_tags=True) == "Uber Raises Delivery Hero Stake in EUR 70 Million Prosus Deal"
    print("Text normalizer test passed")


if __name__ == "__main__":
    main()
