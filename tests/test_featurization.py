import numpy as np
from rdkit import Chem

from clef.featurize import MolFeaturizer
from clef.featurize.isotopes import isotope_template
from clef.msutil.binutils import create_spectrum_bins


def test_physical_spectrum_grid_and_chlorine_isotopes():
    bins = create_spectrum_bins()
    assert bins.mass_to_index(np.asarray([1.0, 512.0, 513.0])).tolist() == [0, 511, -1]
    dense = bins.peaks_to_dense([[1.0, 2.0], [1.0, 3.0], [512.0, 4.0]])
    assert dense[0] == 5.0 and dense[511] == 4.0
    mass_index, intensity = isotope_template(np.asarray([0, 0, 0, 0, 0, 0, 0, 1]))
    assert mass_index[:2].tolist() == [34, 36]
    np.testing.assert_allclose(intensity[:2].sum(), 1.0, rtol=1e-6)


def test_graph_formula_and_event_contract():
    molecule = Chem.MolFromSmiles("CCO")
    featurizer = MolFeaturizer()
    result = featurizer(molecule)
    assert result["adj"].shape == (4, 48, 48)
    assert result["vect_feat"].shape == (48, 45)
    assert result["input_mask"].sum() == 9
    assert result["vect_feat"][0, -1] == 0.75
    assert result["formula_features"].shape == (42, 276)
    assert result["event_features"].shape[1] == 45
    assert result["event_atom_idx"].shape[1] == 4
    assert result["event_features"][0, 0] == 1.0
    assert result["event_atom_idx"][0].tolist() == [-1] * 4
    assert result["event_mass_idx"][0] == 45
    assert np.all(result["event_counts"] <= np.asarray([6, 2, 0, 1, 0, 0, 0, 0]))
    for event, formula_index in zip(result["event_counts"], result["event_formula_index"]):
        np.testing.assert_array_equal(event, result["formula_counts"][formula_index])
    np.testing.assert_allclose(
        result["event_features"][:, 11:19] * result["input_mask"].sum(),
        result["event_counts"],
        atol=1e-6,
    )
    rebuilt = featurizer(molecule)
    np.testing.assert_array_equal(result["event_features"], rebuilt["event_features"])
    np.testing.assert_array_equal(result["event_atom_idx"], rebuilt["event_atom_idx"])


def test_event_precomputation_and_disabled_events():
    molecule = Chem.MolFromSmiles("c1ccccc1")
    featurizer = MolFeaturizer(event_config={"max_events": 32})
    features, masses, endpoints, mask = featurizer._build_events(molecule)
    assert features.shape == (32, 45)
    assert endpoints.shape == (32, 4)
    assert mask.sum() == 32
    assert masses[0] == 78
    assert np.any(features[:, 3] > 0)
    disabled = MolFeaturizer(event_config={"enabled": False})(molecule)
    assert disabled["event_features"].shape == (0, 45)
    assert disabled["event_formula_index"].shape == (0,)
