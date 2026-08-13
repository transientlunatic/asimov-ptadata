import pytest

from asimov_ptadata.fetch import ReleaseNotFoundError, ReleaseSource, fetch_pulsar


@pytest.fixture
def release_dir(tmp_path):
    psr_dir = tmp_path / "J1713+0747"
    psr_dir.mkdir()
    (psr_dir / "J1713+0747.par").write_text("PSR J1713+0747\n")
    (psr_dir / "J1713+0747.NANOGrav.15yr.tim").write_text("FORMAT 1\n")
    (psr_dir / "J1713+0747.EPTA.tim").write_text("FORMAT 1\n")
    (tmp_path / "clock").mkdir()
    (tmp_path / "clock" / "gbt2gps.clk").write_text("")
    return tmp_path


@pytest.fixture
def flat_release_dir(tmp_path):
    (tmp_path / "J1713+0747.par").write_text("PSR J1713+0747\n")
    (tmp_path / "J1713+0747.tim").write_text("FORMAT 1\n")
    return tmp_path


def test_pulsar_dir_locates_directory(release_dir):
    source = ReleaseSource(release_dir)
    assert source.pulsar_dir("J1713+0747") == release_dir / "J1713+0747"


def test_pulsar_dir_none_for_flat_layout(flat_release_dir):
    source = ReleaseSource(flat_release_dir)
    assert source.pulsar_dir("J1713+0747") is None


def test_pulsar_dir_ambiguous_raises(tmp_path):
    (tmp_path / "VersionA" / "J1713+0747").mkdir(parents=True)
    (tmp_path / "VersionB" / "J1713+0747").mkdir(parents=True)
    source = ReleaseSource(tmp_path)
    with pytest.raises(ReleaseNotFoundError):
        source.pulsar_dir("J1713+0747")


def test_par_candidates_prefers_plainest_name(release_dir, tmp_path):
    psr_dir = release_dir / "J1713+0747"
    (psr_dir / "J1713+0747.TDB.par").write_text("PSR J1713+0747\n")

    outdir = tmp_path / "staged"
    manifest = fetch_pulsar("J1713+0747", release_dir, outdir)

    assert manifest["par"].name == "J1713+0747.par"
    assert {p.name for p in manifest["par candidates"]} == {"J1713+0747.par", "J1713+0747.TDB.par"}


def test_par_candidates_prefers_tdb_over_plainest_name(release_dir, tmp_path):
    """PINT refuses to load UNITS TCB par files - checked against a real IPTA
    DR2 pulsar where the plainest-named file was TCB. The plainer filename
    must lose to the TDB one here."""
    psr_dir = release_dir / "J1713+0747"
    (psr_dir / "J1713+0747.par").write_text("PSR J1713+0747\nUNITS TCB\n")
    (psr_dir / "J1713+0747.TDB.par").write_text("PSR J1713+0747\nUNITS TDB\n")

    outdir = tmp_path / "staged"
    manifest = fetch_pulsar("J1713+0747", release_dir, outdir)

    assert manifest["par"].name == "J1713+0747.TDB.par"


def test_clock_dir_found(release_dir):
    source = ReleaseSource(release_dir)
    assert source.clock_dir() == release_dir / "clock"


def test_missing_pulsar_raises(release_dir):
    outdir = release_dir / "staged"
    with pytest.raises(ReleaseNotFoundError):
        fetch_pulsar("J0000+0000", release_dir, outdir)


def test_missing_release_root_raises(tmp_path):
    with pytest.raises(ReleaseNotFoundError):
        ReleaseSource(tmp_path / "does-not-exist")


def test_fetch_pulsar_stages_directory_layout(release_dir, tmp_path):
    outdir = tmp_path / "staged"
    manifest = fetch_pulsar("J1713+0747", release_dir, outdir)

    assert manifest["par"].exists()
    assert len(manifest["tim"]) == 2
    assert all(t.exists() for t in manifest["tim"])
    assert manifest["clock dir"] == release_dir / "clock"


def test_fetch_pulsar_stages_flat_layout(flat_release_dir, tmp_path):
    outdir = tmp_path / "staged"
    manifest = fetch_pulsar("J1713+0747", flat_release_dir, outdir)

    assert manifest["par"].exists()
    assert manifest["par"].parent == outdir
    assert len(manifest["tim"]) == 1


def test_fetch_pulsar_preserves_relative_includes(tmp_path):
    """Real IPTA-DR2-style layout: the top-level .tim INCLUDEs files in a tims/
    subdirectory by relative path, so that subdirectory has to move with it."""
    psr_dir = tmp_path / "J1909-3744"
    (psr_dir / "tims").mkdir(parents=True)
    (psr_dir / "J1909-3744.IPTADR2.par").write_text("PSR J1909-3744\n")
    (psr_dir / "J1909-3744.IPTADR2.tim").write_text("FORMAT 1\nINCLUDE tims/NRT.BON.1400.tim\n")
    (psr_dir / "tims" / "NRT.BON.1400.tim").write_text("FORMAT 1\n")

    outdir = tmp_path / "staged"
    manifest = fetch_pulsar("J1909-3744", tmp_path, outdir)

    staged_tim = manifest["tim"][0]
    included = staged_tim.parent / "tims" / "NRT.BON.1400.tim"
    assert included.exists()
