from barber_label_repair_zxingcpp_v2.pipeline import grade


def row(rotation, geometry, preprocessing, mapping=(1, 2, 3, 0)):
    return {
        "accepted_geometry": True,
        "semantic_to_manual": list(mapping),
        "input_rotation": rotation,
        "geometry_family": geometry,
        "preprocessing_family": preprocessing,
        "valid": True,
        "payload_sha256": "same",
        "mirrored": None,
    }


def test_strict_za_and_conflicting_vote_downgrade():
    rows = []
    for rot in (0, 90, 180, 270):
        rows.append(row(rot, "crop", "gray"))
        rows.append(row(rot, "warp", "unsharp"))
    result, ordering, facts = grade(rows)
    assert result == "ZA"
    assert ordering == (1, 2, 3, 0)
    assert facts["successful_decodes"] == 8
    rows.extend(row(0, "crop", "gray", (0, 1, 2, 3)) for _ in range(3))
    assert grade(rows)[0] != "ZA"
