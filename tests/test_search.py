from cv_agent.search import BASearch


def test_parse_ba():
    data = {
        "ergebnisliste": [
            {
                "referenznummer": "12265-271357_JB5218890-S",
                "stellenangebotsTitel": "AI Engineer",
                "firma": "ACME GmbH",
                "stellenlokationen": [
                    {"adresse": {"plz": "88239", "ort": "Wangen im Allgäu"}}
                ],
                "datumErsteVeroeffentlichung": "2026-08-01",
            }
        ]
    }
    vacs = BASearch._parse(data)
    assert len(vacs) == 1
    assert vacs[0].refnr == "12265-271357_JB5218890-S"
    assert vacs[0].location == "88239 Wangen im Allgäu"
    assert vacs[0].source == "arbeitsagentur"
    assert vacs[0].source_metadata["provider_refnr"] == "12265-271357_JB5218890-S"


def test_parse_ba_empty():
    assert BASearch._parse({}) == []
