"""
Regression defense line for multi-format round-trip conversion.

The library can read and write a number of subtitle formats on top of a single
internal representation (SubStation Alpha). This module guards the *semantic*
stability of converting a subtitle through a format and back:

    SSAFile -> write to format X -> read from format X      (single-format RT)
    SSAFile -> X -> Y -> read back                           (format-pair RT)

MicroDVD is intentionally out of scope (it is frame-based and needs an FPS).
SAMI and Whisper-JAX are read-only formats (they have no writer), so they are
covered by parse-only smoke tests rather than round-trips.

See ``docs/roundtrip-criteria.rst`` for the full criteria, comparison axes and
the rationale for every whitelisted (accepted) difference.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import pytest

from pysubs2 import Alignment, Color, SSAEvent, SSAFile, SSAStyle, make_time
from pysubs2.formats.substation import parse_tags

#: Formats that can be both read and written (MicroDVD excluded by mandate).
READ_WRITE_FORMATS = ["srt", "vtt", "ass", "ssa", "mpl2", "tmp", "ttml", "json"]

#: Read-only formats: no writer exists, parse behavior is smoke-tested only.
READ_ONLY_FORMATS = ["sami", "whisper_jax"]


# ---------------------------------------------------------------------------
# Comparison axis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventAxis:
    """
    The semantic comparison axis for a single visible subtitle event.

    Fields that have no meaning outside SubStation (layer, actor name, margins,
    effect, comments, drawings) are compared only where the target format can
    represent them; see the capability table in the criteria document.

    Text is compared semantically, not by raw override-tag string:

    - ``plaintext`` is the text with all override tags removed;
    - ``styling`` is the sequence of (text-fragment, i/b/u/s style state) after
      running the same tag parser the writers use. Re-serializing nested tags may
      emit extra empty fragments or reorder equivalent markup, so empty fragments
      are ignored and only the styled text spans are compared.
    """
    start: int
    end: int
    style: str
    plaintext: str
    styling: tuple[tuple[str, tuple[bool, bool, bool, bool]], ...]

    @classmethod
    def from_event(cls, event: SSAEvent) -> "EventAxis":
        fragments = parse_tags(event.text, SSAStyle.DEFAULT_STYLE)
        styling = tuple(
            (fragment, (sty.italic, sty.bold, sty.underline, sty.strikeout))
            for fragment, sty in fragments
            if fragment
        )
        return cls(
            start=event.start,
            end=event.end,
            style=event.style,
            plaintext=event.plaintext,
            styling=styling,
        )


def visible_axes(subs: SSAFile) -> list[EventAxis]:
    """Comparison is over visible text events only (comments/drawings excluded)."""
    return [EventAxis.from_event(e) for e in subs.events if e.is_text]


# ---------------------------------------------------------------------------
# Time quantization: documented, pre-existing rounding conventions.
# These are NOT defects and the tolerances encode the format granularity.
# ---------------------------------------------------------------------------

#: Tolerance for START timestamps (ms) introduced by one write/read of a format.
#: SRT/VTT/TTML/JSON keep millisecond precision; SubStation rounds to
#: centiseconds (Aegisub convention, see SubstationFormat.ms_to_timestamp and
#: tests/formats/test_substation.py::test_centisecond_rounding); MPL2 uses
#: 10 ms units; TMP stores whole-second starts.
TIME_TOLERANCE_MS: dict[str, int] = {
    "srt": 0,
    "vtt": 0,
    "ass": 5,
    "ssa": 5,
    "ttml": 0,
    "json": 0,
    "mpl2": 5,
    "tmp": 1000,
}

#: Note on end times: TMP stores no end time at all. It only records a start
#: time and synthesizes the end on read (start + 500ms + 67ms/char, then overlap
#: clamping -- see TmpFormat.from_file). Its end value is regenerated rather than
#: quantized, so the end axis is covered by an explicit whitelist entry instead
#: of a numeric tolerance (see the per-scenario "end" entries below).


def time_tolerance(formats: tuple[str, ...]) -> int:
    """Cumulative tolerance across every format traversed in the chain."""
    return sum(TIME_TOLERANCE_MS[f] for f in formats)


# ---------------------------------------------------------------------------
# Whitelist of accepted (non-defect) semantic differences, each with evidence.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AcceptedDifference:
    formats: frozenset[str]
    scenario: str
    axis: str
    reason: str


def _f(*formats: str) -> frozenset[str]:
    return frozenset(formats)


# Every entry is justified in docs/roundtrip-criteria.rst (same key headings).
# Each entry corresponds to a difference the comparison below actually observes;
# the time-tolerance table (TIME_TOLERANCE_MS) separately documents the accepted
# timing quantization instead of listing it as per-scenario whitelist entries.
WHITELIST: list[AcceptedDifference] = [
    # --- Styling that the target format cannot express (text axis) ---
    AcceptedDifference(_f("mpl2"), "inline_tags", "text",
        "MPL2 writer emits plaintext only; its docstring states 'No styling is "
        "supported' (only reader-side slash-italics exist)."),
    AcceptedDifference(_f("mpl2"), "styles", "text",
        "MPL2 has no style model; styles and inline styling are dropped."),
    AcceptedDifference(_f("ttml"), "styles", "text",
        "TTML parser does not read styling back (docs: advanced styling 'not "
        "supported at all by the parser'); only the writer emits it."),
    AcceptedDifference(_f("srt", "vtt", "tmp"), "styles", "text",
        "Named styles have no storage in these formats; the writer inlines the "
        "style's i/b/u/s state as inline tags, so on read the markup is present "
        "but the event reverts to the Default named style."),

    # --- Named-style axis: lightweight formats have no style storage ---
    AcceptedDifference(_f("srt", "vtt", "mpl2", "tmp", "ttml"), "styles", "style",
        "Non-SubStation formats have no named-style storage; the event reverts to "
        "Default and applicable style attributes are inlined where supported."),

    # --- Tag coverage limits of the lightweight formats ---
    AcceptedDifference(_f("srt", "vtt"), "special_and_unknown_tags", "text",
        "Unknown SubStation override tags (\\an, \\fn, \\k, \\pos...) have no HTML "
        "equivalent and are intentionally dropped (see test_keep_ssa_tags)."),
    AcceptedDifference(_f("mpl2", "tmp", "ttml"), "special_and_unknown_tags", "text",
        "Unknown SubStation override tags have no representation in these formats."),
    AcceptedDifference(_f("ttml"), "inline_tags", "text",
        "TTML parser ignores span styling on read (writer-only styling support)."),

    # --- Whitespace/newline conventions ---
    AcceptedDifference(_f("srt", "vtt", "mpl2", "tmp", "ttml"), "plain", "text",
        "Hard space \\h becomes a literal space and soft newline \\n normalizes to "
        "\\N; an empty line inside an event cannot be represented (SRT/VTT/TMP use "
        "line breaks / '|' as the only separators and collapse blank lines)."),

    # --- SubStation-specific event metadata (metadata axis) ---
    AcceptedDifference(_f("srt", "vtt", "mpl2", "tmp", "ttml"), "ssa_fields", "metadata",
        "Layer, actor name, margins and effect are SubStation-only fields with no "
        "representation in these formats."),
    AcceptedDifference(_f("ass"), "ssa_fields", "metadata",
        "The SSA-only Marked flag has no ASS representation; ASS events carry Layer "
        "instead, so Marked resets on an ASS round-trip."),
    AcceptedDifference(_f("ssa"), "ssa_fields", "metadata",
        "SSA v4 lacks the ASS Layer field (it stores Marked instead); the ASS Layer "
        "value does not survive an SSA round-trip."),

    # --- Comments and drawings (total event count, including hidden events) ---
    AcceptedDifference(_f("srt", "vtt", "mpl2", "tmp", "ttml"), "comments_drawings", "count_total",
        "Non-SubStation writers skip comments and vector drawings "
        "(SSAFile.get_text_events); the events are intentionally absent on read."),

    # --- TMP cannot hold empty-text events ---
    AcceptedDifference(_f("tmp"), "empty_events", "count_total",
        "TMP lines are 'timestamp:text'; an empty text produces no parseable event "
        "(see TMP_LINE / TmpFormat.from_file)."),
]


def is_whitelisted(formats: frozenset[str], scenario: str, axis: str) -> bool:
    # A difference is accepted if any format traversed by the chain declares it,
    # since a limitation of any single link in the chain explains the result.
    return any(
        d.scenario == scenario and d.axis == axis and bool(d.formats & formats)
        for d in WHITELIST
    )


# ---------------------------------------------------------------------------
# Scenario set: deliberately exercises every semantic axis and known edge.
# ---------------------------------------------------------------------------

def _build_plain() -> SSAFile:
    subs = SSAFile()
    subs.append(SSAEvent(start=1000, end=2500, text="Hello, world!"))
    subs.append(SSAEvent(start=2500, end=4000, text=r"line one\Nline two"))
    subs.append(SSAEvent(start=4000, end=5000, text=r"hard\hspace and soft\nnewline"))
    subs.append(SSAEvent(start=5000, end=6000, text="unicode: Příliš žluťoučký 日本国"))
    return subs


def _build_inline_tags() -> SSAFile:
    subs = SSAFile()
    subs.append(SSAEvent(start=1000, end=2000, text=r"plain {\i1}italic{\i0} plain"))
    subs.append(SSAEvent(start=2000, end=3000,
                         text=r"{\b1}bold{\b0} {\u1}underline{\u0} {\s1}strike{\s0}"))
    subs.append(SSAEvent(start=3000, end=4000,
                         text=r"nested {\i1}a {\b1}b{\b0} c{\i0} end"))
    return subs


def _build_styles() -> SSAFile:
    subs = SSAFile()
    fancy = SSAStyle(italic=True, bold=True, underline=True, strikeout=True,
                     primarycolor=Color(r=255, g=0, b=0, a=128),
                     fontname="Comic Sans MS", fontsize=24.0,
                     alignment=Alignment.TOP_LEFT)
    subs.styles["Fancy"] = fancy
    subs.append(SSAEvent(start=1000, end=2000, style="Fancy", text="styled line"))
    subs.append(SSAEvent(start=2000, end=3000, style="Default",
                         text=r"default with {\i1}inline italic{\i0}"))
    return subs


def _build_comments_drawings() -> SSAFile:
    subs = SSAFile()
    subs.append(SSAEvent(start=1000, end=2000, text="visible"))
    comment = SSAEvent(start=1000, end=2000, text="a comment")
    comment.is_comment = True
    subs.append(comment)
    subs.append(SSAEvent(start=2000, end=3000,
                         text=r"{\p1}m 0 0 l 100 0 100 100 0 100{\p0}drawing"))
    return subs


def _build_ssa_fields() -> SSAFile:
    subs = SSAFile()
    subs.append(SSAEvent(start=1000, end=2000, text="layered", layer=3, name="Actor",
                         marginl=7, marginr=8, marginv=9, effect="karaoke", marked=True))
    return subs


def _build_special_tags() -> SSAFile:
    subs = SSAFile()
    subs.append(SSAEvent(start=1000, end=2000, text="Q&A: 5 < 6 & 7 > 4"))
    subs.append(SSAEvent(start=2000, end=3000,
                         text=r"{\an8}positioned {\fnArial}font {\k42}karaoke {\pos(10,20)}x"))
    return subs


def _build_timestamps() -> SSAFile:
    # Millisecond-precision, non-centisecond-aligned values expose rounding.
    subs = SSAFile()
    subs.append(SSAEvent(start=0, end=4, text="four ms"))
    subs.append(SSAEvent(start=1500, end=2505, text="odd ms"))
    subs.append(SSAEvent(start=make_time(h=1), end=make_time(h=1, s=1), text="one hour"))
    return subs


def _build_empty_events() -> SSAFile:
    subs = SSAFile()
    subs.append(SSAEvent(start=1000, end=2000, text=""))
    subs.append(SSAEvent(start=2000, end=3000, text=""))
    return subs


SCENARIOS: dict[str, SSAFile] = {
    "plain": _build_plain(),
    "inline_tags": _build_inline_tags(),
    "styles": _build_styles(),
    "comments_drawings": _build_comments_drawings(),
    "ssa_fields": _build_ssa_fields(),
    "special_and_unknown_tags": _build_special_tags(),
    "timestamps": _build_timestamps(),
    "empty_events": _build_empty_events(),
}

# TMP stores no end time and regenerates it on read (start + 500ms + 67ms/char,
# then overlap clamping). Short captions happen to land within the 1000 ms start
# tolerance anyway; for longer captions the synthesized end exceeds it, so those
# scenarios carry an explicit "end" whitelist entry.
_TMP_END_SCENARIOS = ("plain", "inline_tags", "styles", "special_and_unknown_tags")
WHITELIST.extend(
    AcceptedDifference(_f("tmp"), scenario, "end",
        "TMP stores no end time; the reader regenerates it as start + 500ms + "
        "67ms/char with overlap clamping (TmpFormat.from_file).")
    for scenario in _TMP_END_SCENARIOS
)


def _round_trip(subs: SSAFile, chain: tuple[str, ...]) -> SSAFile:
    current = subs
    with warnings.catch_warnings():
        # Timestamp clamping / missed-subtitle heuristics are expected behavior.
        warnings.simplefilter("ignore")
        for fmt in chain:
            text = current.to_string(fmt)
            current = SSAFile.from_string(text, fmt)
    return current


_METADATA_FIELDS = ("layer", "name", "marginl", "marginr", "marginv", "effect", "marked")


def _metadata(event: SSAEvent) -> tuple:
    return tuple(getattr(event, f) for f in _METADATA_FIELDS)


def _assert_axis_equivalent(chain: tuple[str, ...], scenario: str,
                            original: SSAFile, result: SSAFile) -> None:
    formats = frozenset(chain)

    # Total event-count axis, counting hidden events (comments/drawings) too.
    count_whitelisted = is_whitelisted(formats, scenario, "count_total")
    if len(original.events) != len(result.events):
        assert count_whitelisted, (
            f"total event count changed for {chain} / {scenario}: "
            f"{len(original.events)} -> {len(result.events)} (not whitelisted)")

    expected = visible_axes(original)
    actual = visible_axes(result)

    # Visible-event count must match unless the same count loss is whitelisted
    # (e.g. TMP cannot represent empty-text events at all).
    if len(expected) != len(actual):
        assert count_whitelisted, (
            f"visible event count changed for {chain} / {scenario}: "
            f"{len(expected)} -> {len(actual)}")
        return

    tolerance = time_tolerance(chain)
    for idx, (want, got) in enumerate(zip(expected, actual)):
        # Start axis: compared against the documented quantization tolerance.
        assert abs(want.start - got.start) <= tolerance, (
            f"start drift {want.start} -> {got.start} exceeds tolerance "
            f"{tolerance} ms for {chain} / {scenario} event {idx}")

        # End axis: tolerance for quantizing formats, whitelist for formats that
        # synthesize the end time rather than storing it (TMP).
        if abs(want.end - got.end) > tolerance:
            assert is_whitelisted(formats, scenario, "end"), (
                f"end drift {want.end} -> {got.end} exceeds tolerance "
                f"{tolerance} ms for {chain} / {scenario} event {idx} "
                f"(not whitelisted)")

        # Named-style axis.
        if want.style != got.style:
            assert is_whitelisted(formats, scenario, "style"), (
                f"style changed for {chain} / {scenario} event {idx}: "
                f"{want.style!r} -> {got.style!r} (not whitelisted)")

        # SubStation-only event metadata axis.
        orig_event = original.events[idx]
        if _metadata(orig_event) != _metadata(result.events[idx]):
            assert is_whitelisted(formats, scenario, "metadata"), (
                f"SSA metadata changed for {chain} / {scenario} event {idx} "
                f"(not whitelisted)")

        # Text axis: plaintext plus the i/b/u/s styling of each fragment.
        text_diffs = []
        if want.plaintext != got.plaintext:
            text_diffs.append(
                f"plaintext {want.plaintext!r} -> {got.plaintext!r}")
        if want.styling != got.styling:
            text_diffs.append(
                f"styling {want.styling!r} -> {got.styling!r}")
        if text_diffs:
            assert is_whitelisted(formats, scenario, "text"), (
                f"text changed for {chain} / {scenario} event {idx}: "
                + "; ".join(text_diffs) + " (not whitelisted)")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", READ_WRITE_FORMATS)
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_single_format_roundtrip(fmt: str, scenario: str) -> None:
    """write -> read through one format must preserve all non-whitelisted axes."""
    original = SCENARIOS[scenario]
    result = _round_trip(original, (fmt,))
    _assert_axis_equivalent((fmt,), scenario, original, result)


@pytest.mark.parametrize("src", READ_WRITE_FORMATS)
@pytest.mark.parametrize("dst", READ_WRITE_FORMATS)
@pytest.mark.parametrize("scenario", ["plain", "timestamps"])
def test_format_pair_roundtrip(src: str, dst: str, scenario: str) -> None:
    """
    Cross every ordered format pair (X -> Y -> read). Only the two scenarios that
    every format is fundamentally able to represent are asserted here; richer
    axes are covered per-format in test_single_format_roundtrip with whitelists.
    """
    original = SCENARIOS[scenario]
    result = _round_trip(original, (src, dst))
    _assert_axis_equivalent((src, dst), scenario, original, result)


@pytest.mark.parametrize("fmt", ["json", "ass"])
def test_lossless_formats_preserve_full_semantics(fmt: str) -> None:
    """
    The richest internal representation must survive through the formats that
    claim full fidelity: JSON is the serialized internal model and ASS is the
    native format. Both keep styles, metadata and every event (including a
    comment and a drawing) -- only ASS centisecond time rounding is expected.
    """
    rich = _build_styles()
    rich.append(SSAEvent(start=1000, end=2000, text="meta", layer=2, name="actor"))
    rich.append(_build_comments_drawings().events[1])  # a comment event

    result = _round_trip(rich, (fmt,))

    assert len(result.events) == len(rich.events)
    assert set(result.styles) == set(rich.styles)
    for got, want in zip(result.events, rich.events):
        # All non-time fields (text, style, layer, name, margins, type) survive.
        for field_name in ("text", "style", "layer", "name", "marginl",
                           "marginr", "marginv", "effect", "type", "marked"):
            assert getattr(got, field_name) == getattr(want, field_name), (
                f"{fmt} lost field {field_name}: "
                f"{getattr(want, field_name)!r} -> {getattr(got, field_name)!r}")
        assert abs(got.start - want.start) <= TIME_TOLERANCE_MS[fmt]
        assert abs(got.end - want.end) <= TIME_TOLERANCE_MS[fmt]


@pytest.mark.parametrize("fmt", READ_ONLY_FORMATS)
def test_read_only_formats_parse_without_crash(fmt: str) -> None:
    """Read-only formats have no writer; ensure their parser still yields events."""
    if fmt == "whisper_jax":
        sample = "[00:01.000 -> 00:02.000] hello world\n"
    else:
        sample = ("<SAMI><Body>"
                  "<SYNC Start=0><P>hello"
                  "<SYNC Start=1000><P>world"
                  "</Body></SAMI>")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        subs = SSAFile.from_string(sample, fmt)
    assert len(subs) >= 1
    assert all(e.text for e in subs)
