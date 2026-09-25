import pytest

from asimov_ptadata.scheduling import job_environment_kwargs


def test_htcondor_environment_uses_the_quoted_new_syntax():
    assert job_environment_kwargs({"A": "1", "B": "/x"}) == {"environment": '"A=1 B=/x"'}


def test_htcondor_quotes_values_with_spaces_and_quotes():
    expected = "\"B='/x y' C='it''s'\""
    assert job_environment_kwargs({"B": "/x y", "C": "it's"}) == {"environment": expected}


def test_htcondor_doubles_literal_double_quotes():
    assert job_environment_kwargs({"A": 'a"b'}) == {"environment": '"A=a""b"'}


def test_slurm_environment_extends_the_exported_environment():
    assert job_environment_kwargs({"A": "1", "B": "2"}, "slurm") == {"slurm_export": "ALL,A=1,B=2"}


def test_slurm_refuses_a_comma_in_a_value():
    with pytest.raises(ValueError):
        job_environment_kwargs({"A": "1,2"}, "slurm")


def test_nothing_to_set():
    assert job_environment_kwargs(None) == {}
    assert job_environment_kwargs({}, "slurm") == {}
