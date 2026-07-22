"""Compatibility imports for the renamed SPIReS data writer module."""

from spires_io.spiresdata_writer import SpiresDataWriter, write_spires_data

__all__ = ["SpiresDataWriter", "write_spires_data"]
