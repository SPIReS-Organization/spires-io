from spires_io.modis import (
    MODIS_DEFAULT_BAND_NAMES,
    MODIS_PRODUCT_TO_PLATFORM,
    normalize_modis_band_names,
    resolve_modis_inversion_bands,
)


def test_modis_band_defaults_and_platforms():
    assert MODIS_DEFAULT_BAND_NAMES == ("1", "2", "3", "4", "5", "6", "7")
    assert MODIS_PRODUCT_TO_PLATFORM["MOD09GA"] == "terra"
    assert MODIS_PRODUCT_TO_PLATFORM["MYD09GA"] == "aqua"


def test_normalize_modis_band_names_accepts_plain_and_prefixed_labels():
    assert normalize_modis_band_names(["1", "B2", " b3 "]) == ["1", "2", "3"]


def test_resolve_modis_inversion_bands_defaults_to_bands_1_through_7():
    assert resolve_modis_inversion_bands() == ["1", "2", "3", "4", "5", "6", "7"]
