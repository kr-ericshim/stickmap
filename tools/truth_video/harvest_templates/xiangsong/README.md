# Xiangsong explicit snare parentheses

Four native-resolution median-reader glyphs retain the four pilots' different
rasterization/blur. They were cropped, never enlarged, from the development
targets recorded in `legends/xiangsong.json` (`source`, `anchor`, and native
`reference_spacing`). Original source frames and matched reader targets are in
`.eval-cache/pilot_new/xiangsong/audit-ghost/`.

The existing composite-glyph reader requires correlation >=0.95 and independently
row-centred left/right parenthesis correlations >=0.65. Only filled snare position
5 is calibrated. These thresholds intentionally miss real ghosts. A false flag
is not a verified ordinary/strong stroke.

The disjoint 48-target source/reader audit in `audit-ghost-fresh/` found 24/24
predicted positives with complete parentheses, and 10 real ghosts among 24
negative predictions. It is an assistant within-source visual audit, not
independent human review, complete score recall or acoustic transcription proof.
Two initial visual decisions were corrected after wider source-target inspection;
both initial and contextual versions remain in `audit-ghost/`.

This channel is sealed VAULT test material. No detector/scorer or training use
is authorized.
