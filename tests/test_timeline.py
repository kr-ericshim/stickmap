"""Run the real TypeScript timeline through the installed web compiler."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_timeline_anchors_cursor_to_raw_notes_and_keeps_legacy_mapping():
    web = Path(__file__).parents[1] / "web"
    if not shutil.which("node") or not (web / "node_modules/typescript").exists():
        pytest.skip("Web TypeScript compiler is not installed")
    script = r"""
const fs = require('fs'), vm = require('vm'), ts = require('typescript'), assert = require('node:assert/strict');
const moduleValue = { exports: {} };
const source = ts.transpileModule(fs.readFileSync('src/timeline.ts', 'utf8'), {compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022}}).outputText;
vm.runInNewContext(source, {exports: moduleValue.exports, module: moduleValue});
const { Timeline } = moduleValue.exports;
const bar = {index:0, beats:1, beat_times:[0,.5], subdivisions:[4], tempo:120, notes:[{tick:0,inst:'kick',velocity:90,raw_time:.02},{tick:12,inst:'snare',velocity:90,raw_time:.29}]};
const score = {bars:[bar], divisions:24};
const raw = new Timeline(score);
assert.equal(raw.noteTime(bar,bar.notes[1]),.29);
assert.equal(raw.wholeAt(.02),0);
assert.equal(raw.wholeAt(.29),.125);
assert.equal(raw.wholeAt(.5),.25);
let previous=-1;
for(let t=0;t<=.51;t+=.001){ const whole=raw.wholeAt(t); assert(whole>=previous); previous=whole; }
bar.notes = bar.notes.map(({raw_time,...note})=>note);
const legacy = new Timeline(score);
assert.equal(legacy.wholeAt(.25),.125);
assert.equal(legacy.noteTime(bar,bar.notes[1]),.25);
const early = {...bar, notes:[{tick:0,inst:'kick',velocity:90,raw_time:-.02}]};
assert.equal(new Timeline({...score,bars:[early]}).barStart(0),-.02);
"""
    subprocess.run(["node", "-e", script], cwd=web, capture_output=True, text=True, check=True)
