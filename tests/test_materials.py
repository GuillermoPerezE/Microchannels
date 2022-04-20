from source import materials


def test_aluminum():
    material = materials.Aluminum()
    assert material.k == 237
    assert material.rho == 2707
    assert material.phase == "Solid"


def test_copper():
    material = materials.Copper()
    assert material.k == 401
    assert material.rho == 8945
    assert material.phase == "Solid"
