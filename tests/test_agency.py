from cv_agent.config import Criteria, is_recruiting_agency


def _criteria():
    return Criteria(
        region="x",
        radius_km=50,
        keywords=["Python"],
        exclude_keywords=[],
        min_score=60,
        recruiting_agency_keywords=["Personaldienstleistung", "Arbeitsvermittlung", "Recruiting"],
        recruiting_agencies=["FERCHAU", "engineering people"],
    )


def test_matches_keyword():
    assert is_recruiting_agency("KMS Personaldienstleistungen GmbH", _criteria()) is True
    assert is_recruiting_agency("Jan Schreiner Private Arbeitsvermittlung", _criteria()) is True
    assert is_recruiting_agency("AllatNet Recruiting GmbH", _criteria()) is True


def test_matches_company_list():
    assert is_recruiting_agency("FERCHAU GmbH", _criteria()) is True
    assert is_recruiting_agency("engineering people GmbH", _criteria()) is True


def test_not_agency():
    assert is_recruiting_agency("Airbus Defence and Space GmbH", _criteria()) is False
    assert is_recruiting_agency("Liebherr-Hausgeräte Ochsenhausen GmbH", _criteria()) is False