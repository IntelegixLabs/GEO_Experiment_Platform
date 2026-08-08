"""Focused tests for the streaming Kaggle Amazon catalog mapper."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.amazon_catalog import (
    AMAZON_DATASET_NAME,
    AmazonCatalogHeaderError,
    AmazonCatalogRowError,
    experimental_variant_id,
    iter_amazon_catalog_records,
    iter_batches,
    map_amazon_row,
    parse_inr_amount,
    parse_review_count,
)


SAMPLE_ROW = {
    "name": "Lloyd 1.5 Ton 3 Star Inverter Split AC",
    "main_category": "appliances",
    "sub_category": "Air Conditioners",
    "image": "https://m.media-amazon.com/images/I/31UISB90sYL._AC_UL320_.jpg",
    "link": "https://www.amazon.in/Lloyd-Inverter-Convertible/dp/B0BRKXTSBT/ref=sr_1_4?qid=1679134237&s=kitchen&sr=1-4",
    "ratings": "4.2",
    "no_of_ratings": "2,255",
    "discount_price": "â‚¹32,999",
    "actual_price": "₹58,990",
}


def test_parse_inr_amount_handles_indian_grouping_mojibake_and_ambiguity() -> None:
    assert parse_inr_amount("₹1,29,999.50") == 129_999.5
    assert parse_inr_amount("â‚¹32,999") == 32_999.0
    assert parse_inr_amount("Rs. 1,299") == 1_299.0
    assert parse_inr_amount("₹1,299 - ₹2,499") is None
    assert parse_inr_amount("nan") is None


def test_maps_source_facts_without_inventing_product_claims() -> None:
    record = map_amazon_row(SAMPLE_ROW, source_row_number=2)

    assert record.source_product_key.startswith("amazon-url-sha256:")
    assert record.source_product_id.startswith("AMZ-H-")
    assert record.pair_id == "AMZ-PAIR-B0BRKXTSBT"
    assert record.category == "appliances"
    assert record.price == 32_999.0
    assert record.review_count == 2_255
    assert record.rating == 4.2

    values = record.catalog_product_values(condition="CONTROL")
    assert values["id"] == record.source_product_id
    assert values["discount_price"] == 32_999.0
    assert values["actual_price"] == 58_990.0
    assert values["currency"] == "INR"
    assert values["description"] == ""
    assert values["key_features"] == []
    assert values["source_dataset"] == AMAZON_DATASET_NAME
    assert values["source_row_number"] == 2
    assert values["condition"] == "CONTROL"


def test_source_url_identity_prevents_asin_collisions_and_variants_keep_the_pair_id() -> None:
    alternate_link = dict(SAMPLE_ROW)
    alternate_link["link"] = "https://www.amazon.in/dp/B0BRKXTSBT?tag=tracking-value"
    alternate_link["discount_price"] = "₹33,499"  # A changing price is not a new listing.

    source = map_amazon_row(SAMPLE_ROW, source_row_number=2)
    changed = map_amazon_row(alternate_link, source_row_number=99)

    # The ASIN is a useful matched-listing group, but it cannot be the unique
    # source identity: the dataset can contain more than one source URL for
    # an ASIN.  Canonical URL identity preserves both rows.
    assert changed.source_product_id != source.source_product_id
    assert changed.source_product_key != source.source_product_key
    assert changed.pair_id == source.pair_id
    control_id = experimental_variant_id(source.source_product_id, "CONTROL")
    geo_id = experimental_variant_id(source.source_product_id, "GEO_OPTIMIZED")
    assert control_id.endswith("-CTRL")
    assert geo_id.endswith("-GEO")
    assert control_id != geo_id
    assert source.catalog_product_values(condition="GEO_OPTIMIZED", product_id=geo_id)["pair_id"] == source.pair_id


def test_streaming_iterator_reports_bad_rows_without_loading_or_silently_losing_them() -> None:
    csv_text = (
        ",name,main_category,sub_category,image,link,ratings,no_of_ratings,discount_price,actual_price\n"
        "0,Good kettle,home,Kettles,https://images.example/kettle.jpg,https://www.amazon.in/dp/B012345678,4.5,1,\"₹1,299\",\"₹1,999\"\n"
        "1,,home,Kettles,https://images.example/bad.jpg,https://www.amazon.in/dp/B012345679,4.4,2,\"₹2,299\",\"₹2,999\"\n"
        "2,Good lamp,home,Lighting,https://images.example/lamp.jpg,https://www.amazon.in/dp/B012345670,4.3,3,₹899,\"₹1,299\"\n"
    )
    errors: list[AmazonCatalogRowError] = []
    records = list(iter_amazon_catalog_records(StringIO(csv_text), on_error=errors.append))

    assert [record.name for record in records] == ["Good kettle", "Good lamp"]
    assert len(errors) == 1
    assert errors[0].row_number == 3
    assert parse_review_count("2.5K") == 2_500
    assert list(iter_batches(records, batch_size=1)) == [[records[0]], [records[1]]]


def test_iterator_raises_for_bad_rows_when_no_error_callback_is_provided() -> None:
    csv_text = (
        "name,main_category,sub_category,image,link,ratings,no_of_ratings,discount_price,actual_price\n"
        ",home,Kettles,,,,,,\n"
    )
    with pytest.raises(AmazonCatalogRowError, match="non-empty name"):
        list(iter_amazon_catalog_records(StringIO(csv_text)))


def test_iterator_rejects_a_csv_without_the_required_kaggle_columns() -> None:
    with pytest.raises(AmazonCatalogHeaderError, match="missing required columns"):
        list(iter_amazon_catalog_records(StringIO("name,category\nKettle,home\n")))
