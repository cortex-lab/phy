"""Reassemble the reviewed narration from the archived capture and approved WAV files.

Requires FFmpeg, NumPy and soundfile; does not generate speech or alter Phy.
"""

import argparse
import hashlib
import json
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import soundfile as sf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--capture', type=Path, required=True)
parser.add_argument('--approved-audio', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source = args.capture.resolve()
approved = args.approved_audio.resolve()
out = args.output.resolve()
# Require a fresh directory to preserve existing review artifacts.
out.mkdir(parents=True, exist_ok=False)
meta = json.loads((approved / 'generation.json').read_text())
sr = meta['sample_rate']
fr = meta['frame_rate']
# (sentence, first word, end word, video start, video limit); word zero is Bonjour.
plan = [
    (1, 0, None, 0.35, 5.2),
    (2, 1, None, 5.45, 7.7),
    (3, 1, None, 11.15, 17.2),
    (4, 1, None, 20.6, 27.9),
    (5, 1, None, 28.2, 30.9),
    (6, 1, None, 31.3, 35.1),
    (7, 1, None, 35.4, 39.3),
    (8, 1, None, 39.7, 43.05),
    (9, 1, None, 43.6, 49.15),
    (10, 1, None, 49.45, 52.7),
    (11, 1, None, 53.1, 57.8),
]
output = np.zeros(round(57.9 * sr), dtype=np.float32)
cues = []
alignment = []
last_end = 0
for number, first, last, start, limit in plan:
    info = meta['sentences'][number - 1]
    words = info['transcript']
    shift = info['decoder_timestamp_shift_seconds']
    last = len(words) if last is None else last
    # Read approved audio, not a new synthesis. Preserve Bonjour only once.
    keep_intro = number == 1
    path = approved / (info['with_bonjour'] if keep_intro else info['file'])
    clip_audio, rate = sf.read(path)
    assert rate == sr
    cut = 0 if keep_intro else info['primer_cut_seconds']
    a = 0 if first <= 1 else max(0, words[first][1] / fr - shift - cut - 0.04)
    b = (
        len(clip_audio) / sr
        if last == len(words)
        else max(a, words[last][1] / fr - shift - cut - 0.04)
    )
    clip = clip_audio[round(a * sr) : round(b * sr)].copy()
    duration = len(clip) / sr
    assert duration <= limit - start + 1 / sr, (number, duration, limit - start)
    assert start >= last_end, (number, start, last_end)
    # Full approved sentences only: no internal cuts or fades.
    if first > 1 or last < len(words):
        fade = min(round(0.005 * sr), len(clip) // 2)
        clip[:fade] *= np.linspace(0, 1, fade)
        clip[-fade:] *= np.linspace(1, 0, fade)
    pos = round(start * sr)
    output[pos : pos + len(clip)] = clip
    last_end = start + duration
    text = ' '.join(w[0] for w in words[first:last])
    cues.append({'start': start + 0.02, 'end': start + duration, 'text': text})
    alignment.append(
        {
            'sentence': number,
            'text': text,
            'source_file': path.name,
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_start': a,
            'source_end': b,
            'output_start': start,
            'output_end': last_end,
            'tempo_factor': 1.0,
        }
    )
assert len(cues) == 11
assert all(x['source_start'] == 0 for x in alignment)
assert (
    min(
        alignment[i + 1]['output_start'] - alignment[i]['output_end']
        for i in range(len(alignment) - 1)
    )
    >= 0.4
)
assert ' '.join(c['text'] for c in cues) == (
    'Bonjour. ' + ' '.join(s['text'] for s in meta['sentences'])
)
sf.write(out / 'narration-timed.wav', output, sr)
subprocess.run(
    [
        'ffmpeg',
        '-nostdin',
        '-v',
        'error',
        '-y',
        '-i',
        str(out / 'narration-timed.wav'),
        '-af',
        'loudnorm=I=-18:TP=-2:LRA=7',
        '-ar',
        str(sr),
        str(out / 'narration-timed-normalized.wav'),
    ],
    check=True,
)
subprocess.run(
    [
        'ffmpeg',
        '-nostdin',
        '-v',
        'error',
        '-y',
        '-i',
        str(out / 'narration-timed-normalized.wav'),
        '-c:a',
        'libmp3lame',
        '-b:a',
        '128k',
        str(out / 'narration-timed.mp3'),
    ],
    check=True,
)
(out / 'alignment.json').write_text(
    json.dumps(
        {
            'phrases': alignment,
            'timing': 'Full approved sentences only; no internal edits, no tempo changes; minimum0.4s silence between sentences',
            'publication': 'Local preview only; not published',
        },
        indent=2,
    )
    + '\n'
)


def ass_time(t):
    n = round(t * 100)
    h, n = divmod(n, 360000)
    m, n = divmod(n, 6000)
    s, c = divmod(n, 100)
    return f'{h}:{m:02}:{s:02}.{c:02}'


def srt_time(t):
    n = round(t * 1000)
    h, n = divmod(n, 3600000)
    m, n = divmod(n, 60000)
    s, c = divmod(n, 1000)
    return f'{h:02}:{m:02}:{s:02},{c:03}'


ass = (source / 'review-release-final.ass').read_text()
# Only move the bottom-centred keycaps; nearby pointer Ctrl badges stay put.
ass = ass.replace(',1060)', ',990)').replace(',1084)', ',1014)')
style = 'Style: Narration,DejaVu Sans,32,&H00FFFFFF,&H00FFFFFF,&H00101010,&H60101010,0,0,0,0,100,100,0,0,3,2,0,2,80,80,30,1\n'
ass = ass.replace('[Events]', style + '\n[Events]')
srt = []
for n, cue in enumerate(cues, 1):
    text = '\\N'.join(textwrap.wrap(cue['text'], width=58))
    ass += f'Dialogue: 8,{ass_time(cue["start"])},{ass_time(cue["end"])},Narration,,0,0,0,,{{\\fad(60,90)}}{text}\n'
    srt.append(
        f'{n}\n{srt_time(cue["start"])} --> {srt_time(cue["end"])}\n'
        + '\n'.join(textwrap.wrap(cue['text'], width=58))
        + '\n'
    )
(out / 'preview.ass').write_text(ass)
(out / 'narration.srt').write_text('\n'.join(srt))
timeline = json.loads((source / 'review-release-final.timeline.json').read_text())
items = timeline['items']
filters = []
labels = []
for i, item in enumerate(items):
    label = f'v{i}'
    labels.append(f'[{label}]')
    if item['kind'] == 'source':
        filters.append(
            f'[0:v]trim=start_frame={item["source_start_frame"]}:end_frame={item["source_end_frame"]},setpts=PTS-STARTPTS[{label}]'
        )
    else:
        frame = item['source_frame']
        pad = max(0, item['duration'] - 1 / 30)
        filters.append(
            f'[0:v]trim=start_frame={frame}:end_frame={frame + 1},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={pad:.6f}[{label}]'
        )
ass_path = (
    str(out / 'preview.ass')
    .replace('\\', '\\\\')
    .replace(':', '\\:')
    .replace("'", "\\'")
    .replace(',', '\\,')
)
filters.append(''.join(labels) + f'concat=n={len(labels)}:v=1:a=0,ass={ass_path}[outv]')
subprocess.run(
    [
        'ffmpeg',
        '-nostdin',
        '-y',
        '-v',
        'warning',
        '-i',
        str(source / 'walkthrough.mp4'),
        '-i',
        str(out / 'narration-timed-normalized.wav'),
        '-filter_complex',
        ';'.join(filters),
        '-map',
        '[outv]',
        '-map',
        '1:a:0',
        '-c:v',
        'libx264',
        '-preset',
        'veryfast',
        '-crf',
        '22',
        '-threads',
        '4',
        '-pix_fmt',
        'yuv420p',
        '-r',
        '30',
        '-fps_mode',
        'cfr',
        '-c:a',
        'aac',
        '-b:a',
        '160k',
        '-t',
        '57.9',
        '-movflags',
        '+faststart',
        str(out / 'narrated-preview.mp4'),
    ],
    check=True,
)
print('Preview complete', flush=True)
print(json.dumps(alignment, indent=2), flush=True)
