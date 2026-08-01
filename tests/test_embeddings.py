import sys
from pathlib import Path

import numpy as np
import pytest


# Allow importing package before pip install
sys.path.append(
    str(Path(__file__).resolve().parents[1] / "src")
)


from surgical_embeddings import (
    generate_embeddings,
    save_embeddings,
)


# ==========================================
# Test data
# ==========================================

TEST_INPUT = [
    "appendectomy",
    "laparoscopic surgery",
    "MRI imaging",
]


# ==========================================
# Helper validation
# ==========================================

def check_embedding_output(output, expected_models):

    assert isinstance(output, dict)

    assert "embeddings" in output
    assert "metadata" in output


    embeddings = output["embeddings"]

    assert set(embeddings.keys()) == set(expected_models)


    for model, embedding in embeddings.items():

        assert isinstance(
            embedding,
            np.ndarray
        )

        assert embedding.shape[0] == len(TEST_INPUT)

        assert embedding.ndim == 2


    metadata = output["metadata"]

    assert isinstance(
        metadata,
        dict
    )

    assert "pca_applied" in metadata
    assert "variance_percent" in metadata



# ==========================================
# Model selection tests
# ==========================================

def test_all_models_no_pca():

    embeddings = generate_embeddings(
        TEST_INPUT,
        model_name="all",
        apply_pca=False,
    )


    check_embedding_output(
        embeddings,
        [
            "SapBERT",
            "MiniLM",
            "BGE_Large",
        ],
    )


    assert (
        embeddings["metadata"]["pca_applied"]
        is False
    )



def test_two_models_sapbert_minilm():

    embeddings = generate_embeddings(
        TEST_INPUT,
        model_name=[
            "SapBERT",
            "MiniLM",
        ],
    )

    check_embedding_output(
        embeddings,
        [
            "SapBERT",
            "MiniLM",
        ],
    )



def test_two_models_sapbert_bge():

    embeddings = generate_embeddings(
        TEST_INPUT,
        model_name=[
            "SapBERT",
            "BGE_Large",
        ],
    )

    check_embedding_output(
        embeddings,
        [
            "SapBERT",
            "BGE_Large",
        ],
    )



def test_two_models_minilm_bge():

    embeddings = generate_embeddings(
        TEST_INPUT,
        model_name=[
            "MiniLM",
            "BGE_Large",
        ],
    )

    check_embedding_output(
        embeddings,
        [
            "MiniLM",
            "BGE_Large",
        ],
    )



# ==========================================
# Single model tests
# ==========================================

@pytest.mark.parametrize(
    "model",
    [
        "SapBERT",
        "MiniLM",
        "BGE_Large",
    ],
)
def test_single_model(model):

    embeddings = generate_embeddings(
        TEST_INPUT,
        model_name=model,
    )


    check_embedding_output(
        embeddings,
        [model],
    )



# ==========================================
# Input validation tests
# ==========================================

def test_single_string_input():

    embeddings = generate_embeddings(
        "appendectomy",
        model_name="MiniLM",
    )


    assert (
        embeddings["embeddings"]["MiniLM"].shape[0]
        == 1
    )



def test_txt_file_input(tmp_path):

    input_path = tmp_path / "terms.txt"

    input_path.write_text(
        "\n".join(TEST_INPUT) + "\n"
    )


    embeddings = generate_embeddings(
        str(input_path),
        model_name="MiniLM",
    )


    assert (
        embeddings["embeddings"]["MiniLM"].shape[0]
        == len(TEST_INPUT)
    )



def test_txt_file_input_skips_blank_lines(tmp_path):

    input_path = tmp_path / "terms.txt"

    input_path.write_text(
        "appendectomy\n\n\nlaparoscopic surgery\n"
    )


    embeddings = generate_embeddings(
        str(input_path),
        model_name="MiniLM",
    )


    assert (
        embeddings["embeddings"]["MiniLM"].shape[0]
        == 2
    )



def test_invalid_input():

    with pytest.raises(ValueError):

        generate_embeddings(
            123,
            model_name="MiniLM",
        )



# ==========================================
# PCA tests
# ==========================================

def test_full_pca_single_model():

    original = generate_embeddings(
        TEST_INPUT,
        model_name="MiniLM",
    )


    pca_embeddings = generate_embeddings(
        TEST_INPUT,
        model_name="MiniLM",
        apply_pca=True,
    )


    check_embedding_output(
        pca_embeddings,
        ["MiniLM"],
    )


    assert (
        pca_embeddings["metadata"]["pca_applied"]
        is True
    )


    assert (
        pca_embeddings["metadata"]["variance_percent"]
        is None
    )


    assert (
        pca_embeddings["embeddings"]["MiniLM"].shape[1]
        ==
        original["embeddings"]["MiniLM"].shape[1]
    )



def test_95_percent_pca_single_model():

    embeddings = generate_embeddings(
        TEST_INPUT,
        model_name="SapBERT",
        apply_pca=True,
        variance_percent=95,
    )


    check_embedding_output(
        embeddings,
        ["SapBERT"],
    )


    assert (
        embeddings["metadata"]["variance_percent"]
        == 95
    )


    assert (
        embeddings["embeddings"]["SapBERT"].shape[1]
        < 768
    )



def test_99_percent_pca_all_models():

    embeddings = generate_embeddings(
        TEST_INPUT,
        model_name="all",
        apply_pca=True,
        variance_percent=99,
    )


    check_embedding_output(
        embeddings,
        [
            "SapBERT",
            "MiniLM",
            "BGE_Large",
        ],
    )



def test_pca_selected_models():

    embeddings = generate_embeddings(
        TEST_INPUT,
        model_name=[
            "MiniLM",
            "BGE_Large",
        ],
        apply_pca=True,
        variance_percent=95,
    )


    check_embedding_output(
        embeddings,
        [
            "MiniLM",
            "BGE_Large",
        ],
    )



# ==========================================
# Error handling
# ==========================================

def test_invalid_model():

    with pytest.raises(ValueError):

        generate_embeddings(
            TEST_INPUT,
            model_name="InvalidModel",
        )



@pytest.mark.parametrize(
    "variance",
    [
        0,
        -10,
        101,
    ],
)
def test_invalid_variance_percent(variance):

    with pytest.raises(ValueError):

        generate_embeddings(
            TEST_INPUT,
            model_name="MiniLM",
            apply_pca=True,
            variance_percent=variance,
        )



# ==========================================
# Saving tests
# ==========================================

def test_save_single_model(tmp_path):

    embeddings = generate_embeddings(
        TEST_INPUT,
        model_name="MiniLM",
    )


    save_embeddings(
        embeddings,
        tmp_path,
    )


    filepath = (
        tmp_path /
        "MiniLM_embeddings_original.npz"
    )


    assert filepath.exists()


    loaded = np.load(filepath)


    assert np.array_equal(
        loaded["embeddings"],
        embeddings["embeddings"]["MiniLM"],
    )



def test_save_all_models(tmp_path):

    embeddings = generate_embeddings(
        TEST_INPUT,
        model_name="all",
    )


    save_embeddings(
        embeddings,
        tmp_path,
    )


    expected_files = [
        "SapBERT_embeddings_original.npz",
        "MiniLM_embeddings_original.npz",
        "BGE_Large_embeddings_original.npz",
    ]


    for filename in expected_files:

        assert (
            tmp_path / filename
        ).exists()



def test_saved_multiple_models_match(tmp_path):

    embeddings = generate_embeddings(
        TEST_INPUT,
        model_name=[
            "MiniLM",
            "BGE_Large",
        ],
    )


    save_embeddings(
        embeddings,
        tmp_path,
    )


    for model in [
        "MiniLM",
        "BGE_Large",
    ]:

        filepath = (
            tmp_path /
            f"{model}_embeddings_original.npz"
        )


        loaded = np.load(filepath)


        assert np.array_equal(
            loaded["embeddings"],
            embeddings["embeddings"][model],
        )



def test_save_pca_embeddings(tmp_path):

    embeddings = generate_embeddings(
        TEST_INPUT,
        model_name="MiniLM",
        apply_pca=True,
        variance_percent=95,
    )


    save_embeddings(
        embeddings,
        tmp_path,
    )


    filepath = (
        tmp_path /
        "MiniLM_embeddings_pca_95percent.npz"
    )


    assert filepath.exists()


    loaded = np.load(filepath)


    assert np.array_equal(
        loaded["embeddings"],
        embeddings["embeddings"]["MiniLM"],
    )



def test_save_full_pca_embeddings(tmp_path):

    embeddings = generate_embeddings(
        TEST_INPUT,
        model_name="MiniLM",
        apply_pca=True,
    )


    save_embeddings(
        embeddings,
        tmp_path,
    )


    filepath = (
        tmp_path /
        "MiniLM_embeddings_pca_full.npz"
    )


    assert filepath.exists()


    loaded = np.load(filepath)


    assert np.array_equal(
        loaded["embeddings"],
        embeddings["embeddings"]["MiniLM"],
    )