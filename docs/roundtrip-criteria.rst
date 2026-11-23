Round-Trip Regression Criteria
==============================

This document defines the regression defense line for multi-format subtitle
conversion in pysubs2, the criteria used to judge whether a round-trip preserves
meaning, the discipline for telling real defects from acceptable differences,
and the rationale for every whitelisted entry and every code fix.

The executable form of this contract is
``tests/formats/test_roundtrip.py``; the whitelist there and the headings in
this document are kept in sync.

1. Scope
--------

The library represents every subtitle internally as a SubStation Alpha
document (``SSAFile`` of ``SSAEvent``/``SSAStyle``). Each external format is a
projection of that model through a reader and a writer. A round-trip exercises
both directions:

- **Single-format round-trip** -- ``SSAFile -> write X -> read X``.
- **Format-pair round-trip** -- ``SSAFile -> write X -> read X -> write Y -> read Y``.

Formats that can be both read and written are covered:

``srt``, ``vtt``, ``ass``, ``ssa``, ``mpl2``, ``tmp``, ``ttml``, ``json``.

Exclusions:

- **MicroDVD** is out of scope by mandate (frame-based, requires an FPS and a
  separate conversion path).
- **SAMI** and **Whisper-JAX** have no writer (``to_file`` raises
  ``NotImplementedError``); they cannot round-trip and are covered by a
  parse-only smoke test instead.

2. Shape of the defense line
----------------------------

Three layers, all in ``tests/formats/test_roundtrip.py``:

1. **Single-format matrix** -- every read-write format crossed with every
   scenario (8 formats x 8 scenarios). This is where format-specific capability
   limits are asserted and whitelisted.
2. **Ordered format-pair matrix** -- every ordered pair (8 x 8 = 64) over the
   two scenarios every format is fundamentally able to carry (plain text and
   timestamps). This guards the transcoding chain itself and ensures no pair
   raises or silently corrupts universally-representable content.
3. **High-fidelity anchors** -- JSON (the serialized internal model) and ASS
   (the native format) must preserve the richest document, plus read-only
   parser smoke tests.

The pair matrix deliberately uses only the universal scenarios: demanding
named styles or SubStation-only fields through, say, MPL2 would be testing the
format's known limitations 64 times rather than the conversion logic. The
richer axes are covered once per format in layer 1 against the whitelist.

3. Scenario set
---------------

Each scenario targets a distinct semantic axis or known edge:

=========================  =====================================================
Scenario                   What it exercises
=========================  =====================================================
``plain``                  Plain/multiline text, hard space ``\h``, soft
                           newline ``\n``, hard newline ``\N``, Unicode.
``inline_tags``            Inline bold/italic/underline/strikeout, including
                           nested overrides.
``styles``                 A named custom style (font, color, alignment,
                           i/b/u/s) plus an inline override.
``comments_drawings``      A comment event and a vector-drawing event.
``ssa_fields``             SubStation-only event fields: layer, actor name,
                           margins, effect, marked.
``special_and_unknown_tags``  XML/HTML-significant characters and override
                           tags with no target equivalent (``\an``, ``\fn``,
                           ``\k``, ``\pos``).
``timestamps``             Millisecond and sub-centisecond values that expose
                           each format's time granularity and rounding.
``empty_events``           Zero-length text events.
=========================  =====================================================

4. Comparison axes and judgement criteria
-----------------------------------------

Five axes are compared. The first counts all events (including hidden ones),
the rest apply to the **visible text events** produced by
``SSAFile.get_text_events`` (comments and drawings are excluded by design from
non-SubStation output):

1. **Total event count** (``count_total``) -- all events including comments and
   vector drawings, so that a format silently dropping a hidden event is still
   observed. A separate visible-count check then runs; a whitelisted total-count
   loss (e.g. TMP cannot hold empty-text events) also covers the visible count.
2. **Start time** -- compared against an explicit per-format tolerance, because
   time granularity is a documented property of each format rather than a bug.
3. **End time** -- same tolerance, except formats that synthesize the end time
   instead of storing it (TMP); those are covered by an explicit whitelist entry
   rather than a numeric tolerance.
4. **Named style** (``style``) -- the style name referenced by the event.
5. **SSA metadata** (``metadata``) -- the SubStation-only event fields layer,
   actor name, margins, effect and the SSA ``marked`` flag.
6. **Text semantics** (``text``) -- deliberately *not* the raw override-tag
   string:
   - ``plaintext`` (tags removed), and
   - the sequence of (text-fragment, i/b/u/s state) produced by the same
     ``parse_tags`` engine the writers use, ignoring empty fragments.

The text-semantic comparison is the key piece of discipline. Re-serializing
nested markup can legitimately change the literal tag string -- for example
``nested {\i1}a {\b1}b{\b0} c{\i0}`` is re-emitted with additional reset tags
-- while rendering identically. Requiring literal equality would flag
equivalent markup as failures and encourage weakening the suite; comparing the
parsed fragment/style sequence judges what the viewer actually sees.

Every whitelist entry maps to an axis name actually emitted by the comparison
(``count_total``, ``end``, ``style``, ``metadata`` or ``text``), and a check in
the test module confirms each entry is triggered by at least one real
round-trip difference, so the whitelist cannot accumulate dead justifications.

A difference on any axis is a failure **unless** a whitelist entry declares it
for that format and scenario and axis. A difference is whitelisted only when it
is explained by a documented format limitation, an explicit pre-existing
convention, or an existing test/history commitment.

5. Time-tolerance table (pre-existing conventions, not changed)
---------------------------------------------------------------

These tolerances encode each format's representable time unit. They are
intentionally generous enough to admit the documented rounding but tight enough
to catch a genuine timestamp bug.

==========  ============  ======================================================
Format      Tolerance     Basis
==========  ============  ======================================================
``srt``     0 ms          Millisecond precision.
``vtt``     0 ms          Millisecond precision.
``ttml``    0 ms          Millisecond precision.
``json``    0 ms          Exact internal representation.
``ass``     5 ms          Centisecond precision with Aegisub rounding; see
``ssa``     5 ms          ``SubstationFormat.ms_to_timestamp`` and
                          ``test_centisecond_rounding`` (4 ms -> 0, 5 ms -> 10).
``mpl2``    5 ms          Timestamps are deciseconds (10 ms units).
``tmp``     1000 ms       Whole-second **start** granularity.
==========  ============  ======================================================

For a chain, tolerances sum across every format traversed. The centisecond
rounding convention is explicitly **not** treated as a defect and was not
modified.

TMP does not store an end time at all. Its reader synthesizes the end as
``start + 500 ms + 67 ms/char`` and then clamps overlaps, so for longer
captions the end differs by more than the 1000 ms start tolerance. Rather than
inflate the tolerance to an arbitrary content-dependent number, the TMP end
axis is whitelisted per scenario (the ``end`` entries). Short TMP captions
whose synthesized end happens to stay within the tolerance need no entry, and
the test module's dead-entry check guarantees no whitelist row is redundant.

6. Discipline for classifying a difference
------------------------------------------

Every observed difference was investigated with a throwaway probe before being
admitted to the suite. A difference is a **real defect** only when all hold:

1. The target format is *capable* of expressing the information (per its own
   spec and the library's documented support).
2. The library already handles the symmetric direction or a sibling feature, so
   the gap is an inconsistency rather than an intentional limitation.
3. No existing test or documented convention commits to the lossy behavior.

Anything failing one of those is a **documented limitation** and goes on the
whitelist with a pointer to the evidence (source docstring, ``docs/supported-
formats.rst``, or an existing test). Fixing a whitelisted limitation would mean
changing a promised behavior. Where a genuine fix did contradict an existing
test, that single test is updated in lockstep and recorded with its provenance
in section 8; every other pre-existing test continues to pass unchanged.

7. Whitelist (accepted differences, with evidence)
--------------------------------------------------

**Time: end-time synthesis (axis ``end``)**

- ``tmp`` -- ``plain``, ``inline_tags``, ``styles`` and
  ``special_and_unknown_tags``: TMP stores no end time; the reader regenerates
  it (``TmpFormat.from_file``). These are exactly the scenarios whose end drift
  exceeds the 1000 ms start tolerance; the rest stay within it. Start-time
  quantization for every format is handled by the tolerance table instead.

**Styling the target cannot express (axis ``text``)**

- ``mpl2`` -- ``inline_tags``/``styles``: the writer docstring states "No
  styling is supported"; it writes ``plaintext`` only (the reader's leading-
  slash italics are a one-way parse feature).
- ``ttml`` -- ``styles``/``inline_tags``: per ``docs/supported-formats.rst``
  the TTML *parser* supports no styling at all; only the writer emits it.
- ``srt``/``vtt``/``tmp`` -- ``styles``: the named style is inlined as inline
  tags, so markup survives but the named-style reference does not.

**Named-style storage (axis ``style``)**

- ``srt``/``vtt``/``mpl2``/``tmp``/``ttml`` -- ``styles``: no named-style
  storage; the event reverts to ``Default`` and supported attributes are
  inlined. (ASS/SSA/JSON keep the style name, so they are absent here.)

**Tag coverage limits (axis ``text``)**

- ``srt``/``vtt`` -- ``special_and_unknown_tags``: SubStation-only overrides
  (``\an``, ``\fn``, ``\k``, ``\pos`` ...) have no HTML equivalent and are
  intentionally dropped; the escape hatch is the existing ``keep_ssa_tags``
  option (see ``test_keep_ssa_tags``).
- ``mpl2``/``tmp``/``ttml`` -- same scenario, same reason for their tag sets.

**Whitespace and newline conventions (axis ``text``)**

- ``srt``/``vtt``/``mpl2``/``tmp``/``ttml`` -- ``plain``: ``\h`` becomes a
  literal space and ``\n`` normalizes to ``\N``; an intentionally blank line
  inside an event cannot be represented because these formats use line breaks
  (or ``|`` in TMP) as the sole separator and the readers collapse blank lines.

**SubStation-only event metadata (axis ``metadata``)**

- Non-SubStation formats -- ``ssa_fields``: layer, actor, margins and effect
  have no representation.
- ``ass`` -- ``ssa_fields``: the SSA-only ``marked`` flag has no ASS
  representation, so it resets.
- ``ssa`` -- ``ssa_fields``: SSA v4 has no ASS Layer field (it stores Marked
  instead), so the layer value is lost.

**Comments, drawings and empty events (axis ``count_total``)**

- All non-SubStation formats -- ``comments_drawings``: writers call
  ``get_text_events`` and skip comments/drawings by design.
- ``tmp`` -- ``empty_events``: a TMP line is ``timestamp:text``; empty text
  yields no parseable event (``TMP_LINE``).

8. Real defects found and fixed
-------------------------------

Three genuine round-trip defects (across four format implementations) were
identified with the discipline above and fixed; each has a dedicated regression
test that fails on the pre-fix code and passes afterward (verified by
temporarily reverting the library changes).

1. **SRT/VTT writer dropped bold** (``formats/subrip.py``). The reader already
   converts ``<b>`` to ``{\b1}`` and the writer emitted italic/underline/
   strikeout, but bold was missing from the writer, so bold was silently lost on
   any SRT/VTT round-trip. Added bold to the styled-fragment output and updated
   the docstring. Tests: ``test_subrip.py::test_bold_survives_roundtrip``,
   ``test_bold_style_emitted_in_srt``, ``test_vtt.py::test_bold_survives_roundtrip``.

2. **TMP could not round-trip bold/italic/strikeout** (``formats/tmp.py``). The
   writer emits ``<b>``/``<i>``/``<u>``/``<s>``, but the reader only recognized
   ``<u>`` (bold, italic and strikeout were stripped). Completed the reader's
   tag mapping for all four and added bold to the writer, so its documented
   "italic, underline and strikeout" (now also bold) styling round-trips.
   Test: ``test_tmp.py::test_styling_survives_roundtrip``.

3. **TTML introduced duplicate spaces on round-trip** (``formats/ttml.py``).
   ``ET.indent`` inserted newline/indentation between inline ``<span>`` siblings;
   the parser's whitespace collapsing turned that, together with spaces already
   at span edges, into doubled inter-word spaces. The parser now treats a
   whitespace-only element separator as a single space, preserves interior
   spacing, and normalizes any resulting run of spaces to one. The Apple Music
   TTML fixture (word-level spans whose separators are pure whitespace) still
   parses unchanged.
   Test: ``test_ttml.py::test_inline_styling_roundtrip_no_duplicate_spaces``.

Existing tests updated in lockstep (commit-history/test conflicts, per the
"whitelist with provenance" rule):

- ``test_tmp.py::test_styled_read`` previously asserted that ``<b>`` and other
  supported inline tags were *dropped* on read. That assertion encoded the very
  defect fixed in item 2, so it was updated to assert the tags now survive
  (the unknown ``<xxx>`` tag is still stripped). This is the only case where an
  existing test's expectation changed; every other pre-existing test passes
  unmodified.

9. Running the suite
--------------------

From the repository root on Windows native Python 3.12 (the package is
installed editable)::

    python -m pytest

The round-trip module alone::

    python -m pytest tests/formats/test_roundtrip.py
