import pytest

from asimov_ptadata.fetch import (
    ReleaseNotFoundError,
    ReleaseSource,
    fetch_pulsar,
    normalisation_notes,
    normalise_tim_line,
)


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


# -- .tim normalisation (tempo2 conventions PINT reads differently) ----------

TOA = "c015621.align.pazr.30min 1419.557 51849.5401158655181   0.252  g   -pta EPTA -group EFF.EBPP.1410\n"


@pytest.mark.parametrize(
    "line",
    [
        "C???? " + TOA,
        "C" + TOA,
        "C200404522.bb 1380.000 53202.6984954393610 1.6 wsrt -i puma1\n",
    ],
)
def test_tempo2_style_commented_toa_becomes_a_pint_comment(line):
    new, commented, dropped, addsat = normalise_tim_line(line)
    assert new == "C " + line
    assert commented and dropped == [] and addsat is None


@pytest.mark.parametrize(
    "line",
    [
        TOA,  # lower-case c is an archive name, not a comment, to tempo2
        "C " + TOA,
        "CC ?c0 1419.557 51849.54 0.252 g\n",
        "# " + TOA,
        "FORMAT 1\n",
        "INCLUDE tims/NRT.BON.1400.tim\n",
        "MODE 1\n",
        "\n",
    ],
)
def test_lines_pint_already_reads_like_tempo2_are_untouched(line):
    assert normalise_tim_line(line) == (line, False, [], None)


def test_valueless_flags_are_dropped():
    line = "m2005.rf 1403.775 53430.916 0.901 pks -f MULTI -projid -beconfig -odd -snr 70.99 -padd -0.085 -end\n"
    new, commented, dropped, _ = normalise_tim_line(line)
    assert new == "m2005.rf 1403.775 53430.916 0.901 pks -f MULTI -snr 70.99 -padd -0.085\n"
    assert not commented
    assert dropped == ["-projid", "-beconfig", "-odd", "-end"]


def test_normalised_lines_parse_in_pint_as_tempo2_reads_them():
    from pint.toa import _parse_TOA_line

    commented, _, _, _ = normalise_tim_line("C???? " + TOA)
    assert _parse_TOA_line(commented)[1]["format"] == "Comment"

    flagged, _, _, _ = normalise_tim_line(TOA.rstrip("\n") + " -projid -snr 70.99 -gof 1.1\n")
    flags = _parse_TOA_line(flagged, fmt="Tempo2")[1]
    assert flags["snr"] == "70.99" and flags["gof"] == "1.1" and "projid" not in flags


def test_fetch_pulsar_normalises_staged_includes_not_the_release(tmp_path):
    psr_dir = tmp_path / "release" / "J1744-1134"
    (psr_dir / "tims").mkdir(parents=True)
    (psr_dir / "J1744-1134.par").write_text("PSR J1744-1134\n")
    (psr_dir / "J1744-1134.tim").write_text("FORMAT 1\nINCLUDE tims/EFF.EBPP.1410.tim\n")
    original = "FORMAT 1\n" + TOA + "C???? " + TOA + TOA.rstrip("\n") + " -projid\n"
    (psr_dir / "tims" / "EFF.EBPP.1410.tim").write_text(original)

    manifest = fetch_pulsar("J1744-1134", tmp_path / "release", tmp_path / "staged")

    summary = manifest["tim normalisation"]
    assert summary["commented TOAs"] == 1
    assert summary["valueless flags dropped"] == 1
    staged = (tmp_path / "staged" / "J1744-1134" / "tims" / "EFF.EBPP.1410.tim").read_text()
    assert "C C???? " in staged and "-projid" not in staged
    assert summary["files changed"] == [tmp_path / "staged" / "J1744-1134" / "tims" / "EFF.EBPP.1410.tim"]
    assert (psr_dir / "tims" / "EFF.EBPP.1410.tim").read_text() == original
    assert len(normalisation_notes(summary)) == 2


def test_clean_release_is_staged_byte_for_byte(release_dir, tmp_path):
    manifest = fetch_pulsar("J1713+0747", release_dir, tmp_path / "staged")
    assert manifest["tim normalisation"] == {
        "commented TOAs": 0, "valueless flags dropped": 0, "addsat corrections": 0, "files changed": []
    }
    assert normalisation_notes(manifest["tim normalisation"]) == []


@pytest.mark.parametrize(
    "flags, expected, addsat",
    [
        ("-pta EPTA -addsat -1", "-pta EPTA -to -1.0", -1.0),
        ("-addsat +1 -pta EPTA", "-pta EPTA -to 1.0", 1.0),
        ("-to -0.897e-6 -addsat -1 -pta EPTA", "-to -1.000000897 -pta EPTA", -1.0),
        ("-pta EPTA -addsat +0", "-pta EPTA", 0.0),
    ],
)
def test_addsat_becomes_a_to_time_offset(flags, expected, addsat):
    base = "c058575.align.pazr.30min 1353.499 56178.827511671731325 0.5 g"
    new, commented, dropped, applied = normalise_tim_line(f"{base} {flags}\n")
    assert new == f"{base} {expected}\n"
    assert applied == addsat and not commented and dropped == []


def test_pint_applies_the_converted_addsat_like_tempo2(tmp_path):
    """tempo2 moves a TOA by -addsat seconds (checked: -addsat -1 moves the
    SAT by exactly 1.0000 s); after conversion PINT must move it the same."""
    import shutil

    import pint.config
    import pint.models
    import pint.toa

    par = tmp_path / "example.par"
    shutil.copy(pint.config.examplefile("NGC6440E.par"), par)
    # A tempo2-format TOA at the barycentre ("@"), so no clock files are needed;
    # PINT still applies -to there.
    toa_line = "fake.ar 1949.609 53478.2858714192189 21.71 @"
    model = pint.models.get_model(str(par))

    def mjd(flags):
        tim = tmp_path / "one.tim"
        tim.write_text("FORMAT 1\n" + normalise_tim_line(f"{toa_line} {flags}\n")[0])
        return pint.toa.get_TOAs(str(tim), model=model).table["mjd"][0]

    shift_s = (mjd("-addsat -1") - mjd("-pta X")).to_value("s")
    assert shift_s == pytest.approx(-1.0, abs=1e-9)


def test_addsat_corrections_are_noted():
    notes = normalisation_notes({"commented TOAs": 0, "valueless flags dropped": 0, "addsat corrections": 2})
    assert notes == ["2 -addsat arrival-time correction(s) were converted to -to time offsets while staging"]
