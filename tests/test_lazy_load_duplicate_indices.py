"""Test regression for H5 reopen/lazy-load overlap bug with duplicate profile_index.

This test verifies the fix for the issue where reopening a combined HDF5 file
and toggling a previously-disabled series caused it to overlap another series' data.
The root cause was that lazy loading used source-local profile_index values instead
of collection position indices when loading from combined files.
"""

import numpy as np

from linak.analysis.density import load_density_profiles_by_index
from linak.storage.hdf5_utils import (
    read_linak_hdf5_profile_headers,
    write_linak_hdf5_profile_collection,
)


def test_lazy_load_duplicate_indices_no_overlap(tmp_path):
    """Test that lazy loading works correctly with combined files having duplicate source indices.

    This reproduces the bug scenario where:
    1. A combined HDF5 file is created with 8 density profiles (4 from each source)
    2. Each source has profiles with metadata profile_index: 0,1,2,3
    3. When combined, the file has duplicate indices in metadata: [0,1,2,3,0,1,2,3]
    4. The fix ensures that reading the combined file correctly uses collection position
       indices [0,1,2,3,4,5,6,7] instead of the source metadata indices
    """
    combined_file = tmp_path / "combined_density.h5"

    # Create profiles with duplicate species and distinct data values
    # Source 1: Au, H, O, H2O with density values starting at [0.1, 0.2, 0.3, 0.4]
    # Source 2: Au, H, O, H2O with density values starting at [0.5, 0.6, 0.7, 0.8]
    source_1_species = ["Au", "H", "O", "H2O"]
    source_2_species = ["Au", "H", "O", "H2O"]

    profiles = []
    bin_centers = np.asarray([0.5, 1.5, 2.5], dtype=float)

    # Source 1 profiles - base values 0.1, 0.2, etc.
    for idx, species in enumerate(source_1_species):
        density = np.asarray([0.1 + idx * 0.1, 0.2 + idx * 0.1, 0.3 + idx * 0.1], dtype=float)
        profiles.append(
            {
                "datasets": {
                    "bin_centers_A": bin_centers.copy(),
                    "density": density,
                    "number_density": density * 1e-23,  # Approximate conversion
                },
                "metadata": {
                    "species": species,
                    "profile_index": idx,  # Source-local index
                    "source_profile_index": idx,
                    "bin_width_A": 1.0,
                },
            }
        )

    # Source 2 profiles - base values 0.5, 0.6, etc. (clearly different from source 1)
    for idx, species in enumerate(source_2_species):
        density = np.asarray([0.5 + idx * 0.1, 0.6 + idx * 0.1, 0.7 + idx * 0.1], dtype=float)
        profiles.append(
            {
                "datasets": {
                    "bin_centers_A": bin_centers.copy(),
                    "density": density,
                    "number_density": density * 1e-23,  # Approximate conversion
                },
                "metadata": {
                    "species": species,
                    "profile_index": idx,  # Source-local index (duplicated!)
                    "source_profile_index": idx,
                    "bin_width_A": 1.0,
                },
            }
        )

    # Write combined file
    write_linak_hdf5_profile_collection(
        combined_file,
        analysis="density",
        profiles=profiles,
        metadata={"source": "test-combined"},
    )

    # Read headers back - should now have correct collection position indices
    headers = read_linak_hdf5_profile_headers(combined_file, expected_analysis="density")

    # Verify headers have correct collection position indices [0-7]
    assert len(headers) == 8
    for idx, header in enumerate(headers):
        assert header["profile_index"] == idx, (
            f"Header {idx} should have profile_index={idx}, got {header['profile_index']}"
        )
        # verify source_profile_index was preserved
        assert header["source_profile_index"] == idx % 4

    # Load profiles using the collection indices
    profiles_loaded = load_density_profiles_by_index(combined_file, list(range(8)))

    assert len(profiles_loaded) == 8

    # Verify profiles are distinct and don't overlap
    # Expected: profiles 0-3 from source 1, profiles 4-7 from source 2
    for idx in range(4):
        # Profiles 0-3 should have density starting with 0.1, 0.2, 0.3, 0.4
        expected_value = 0.1 + idx * 0.1
        loaded_value = float(profiles_loaded[idx].density[0])
        assert abs(loaded_value - expected_value) < 0.01, (
            f"Profile {idx} has incorrect data: expected ~{expected_value}, "
            f"got {loaded_value} (overlap bug!)"
        )

    for idx in range(4, 8):
        # Profiles 4-7 should have density starting with 0.5, 0.6, 0.7, 0.8
        expected_value = 0.5 + (idx - 4) * 0.1
        loaded_value = float(profiles_loaded[idx].density[0])
        assert abs(loaded_value - expected_value) < 0.01, (
            f"Profile {idx} has incorrect data: expected ~{expected_value}, got {loaded_value}"
        )

    # Verify that profile 0 and profile 4 are different (they shouldn't overlap)
    assert not np.allclose(
        profiles_loaded[0].density,
        profiles_loaded[4].density,
    ), "Profile 0 and 4 should have different data - overlap bug!"

    # Verify the data difference is substantial
    max_diff = np.max(np.abs(profiles_loaded[0].density - profiles_loaded[4].density))
    assert max_diff > 0.3, f"Profile 0 and 4 should differ significantly, max diff={max_diff}"


def test_lazy_load_maintains_species_correctness(tmp_path):
    """Verify that species labels don't get mixed up due to index aliasing."""
    combined_file = tmp_path / "combined_species.h5"

    # Create distinct species from each source
    profiles = []
    bin_centers = np.asarray([0.5, 1.5], dtype=float)

    # Source 1: unique species A, B
    for idx, species in enumerate(["A", "B"]):
        density = np.asarray([float(idx + 1), float(idx + 1)], dtype=float)
        profiles.append(
            {
                "datasets": {
                    "bin_centers_A": bin_centers.copy(),
                    "density": density,
                    "number_density": density * 1e-23,
                },
                "metadata": {
                    "species": species,
                    "profile_index": idx,
                    "source_profile_index": idx,
                    "bin_width_A": 1.0,
                },
            }
        )

    # Source 2: different species C, D (not A, B from source 1)
    for idx, species in enumerate(["C", "D"]):
        density = np.asarray([float(idx + 10), float(idx + 10)], dtype=float)
        profiles.append(
            {
                "datasets": {
                    "bin_centers_A": bin_centers.copy(),
                    "density": density,
                    "number_density": density * 1e-23,
                },
                "metadata": {
                    "species": species,
                    "profile_index": idx,
                    "source_profile_index": idx,
                    "bin_width_A": 1.0,
                },
            }
        )

    write_linak_hdf5_profile_collection(
        combined_file,
        analysis="density",
        profiles=profiles,
        metadata={"source": "test-species"},
    )

    # Read and verify species correctness
    headers = read_linak_hdf5_profile_headers(combined_file, expected_analysis="density")

    expected_species = ["A", "B", "C", "D"]
    for idx, header in enumerate(headers):
        assert header["species"] == expected_species[idx], (
            f"Profile {idx} should have species '{expected_species[idx]}', "
            f"got '{header['species']}'"
        )
