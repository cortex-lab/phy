"""Add prominent keycaps and pointer-adjacent modifiers to a GUI recording.

The scientific picture is preserved below a 120px input header. A small Ctrl
badge accompanies modified clicks beside the pointer. Frame timing is retained.
"""

import argparse
import json
import subprocess
from pathlib import Path


def timestamp(seconds):
    ticks = round(seconds * 100)
    seconds, fraction = divmod(ticks, 100)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}:{minutes:02}:{seconds:02}.{fraction:02}'


def rounded_box(width, height, radius=10):
    return (
        f'm {radius} 0 l {width - radius} 0 '
        f'b {width} 0 {width} 0 {width} {radius} l {width} {height - radius} '
        f'b {width} {height} {width} {height} {width - radius} {height} '
        f'l {radius} {height} b 0 {height} 0 {height} 0 {height - radius} '
        f'l 0 {radius} b 0 0 0 0 {radius} 0'
    )


def render(directory, output_name=None):
    directory = directory.resolve()
    report = json.loads((directory / 'capture.json').read_text())
    assert report['success'] and report['source_unchanged']
    details = json.loads(
        subprocess.check_output(
            [
                'ffprobe',
                '-v',
                'error',
                '-show_streams',
                '-show_format',
                '-of',
                'json',
                str(directory / 'walkthrough.mp4'),
            ],
            text=True,
        )
    )
    video = next(stream for stream in details['streams'] if stream['codec_type'] == 'video')
    duration = float(details['format']['duration'])
    events = [
        {'seconds': 0, 'label': 'Real data walkthrough on a private display'},
        *report['input_events'],
    ]
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video['width']}
PlayResY: {video['height'] + 120}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Input,DejaVu Sans,26,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,24,24,0,1
Style: Key,DejaVu Sans,40,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for index, event in enumerate(events):
        start = float(event['seconds'])
        stop = float(events[index + 1]['seconds']) if index + 1 < len(events) else duration
        label = event['label'].replace('\n', ' ').replace('{', '').replace('}', '')
        lines.append(
            f'Dialogue: 0,{timestamp(start)},{timestamp(stop)},Input,,0,0,0,,{{\\pos({video["width"] // 2},101)}}{label}'
        )

        if 'click' in label.lower():
            keys = ['Ctrl'] if label.startswith('Ctrl + ') else []
            keys.append('Right click' if label.startswith('Right click') else 'Left click')
            pressed = float(event.get('click_seconds', start + 0.9))
            released = pressed + 0.25
        elif ':' in label and label.split(':', 1)[0] in (
            'G',
            'Ctrl + Z',
            'Ctrl + Shift + Z',
            'Ctrl + S',
            'Enter',
        ):
            keys = label.split(':', 1)[0].split(' + ')
            pressed = float(event.get('key_seconds', start + 0.45))
            released = float(event.get('key_released_seconds', pressed + 0.1))
        else:
            continue
        cue_start = max(0, pressed - 0.18)
        cue_stop = min(duration, max(released + 1.0, cue_start + 1.5))
        widths = [max(78, len(key) * 24 + 34) for key in keys]
        x = (video['width'] - sum(widths) - 32 * (len(keys) - 1)) // 2
        for key_index, (key, width) in enumerate(zip(keys, widths)):
            timing = f'{timestamp(cue_start)},{timestamp(cue_stop)}'
            lines.append(
                f'Dialogue: 1,{timing},Key,,0,0,0,,{{\\an7\\pos({x},14)\\fad(70,180)\\p1\\bord2\\3c&H4DCCFF&\\c&H35414D&}}{rounded_box(width, 62)}'
            )
            lines.append(
                f'Dialogue: 2,{timing},Key,,0,0,0,,{{\\pos({x + width // 2},44)\\fad(70,180)}}{key}'
            )
            x += width
            if key_index + 1 < len(keys):
                lines.append(
                    f'Dialogue: 2,{timing},Key,,0,0,0,,{{\\pos({x + 16},44)\\fad(70,180)}}+'
                )
                x += 32
        if keys[0] == 'Ctrl' and 'point' in event:
            px, py = event['point']
            bx = min(max(px + 38, 4), video['width'] - 112)
            by = max(124, py + 120 - 54)
            lines.append(
                f'Dialogue: 3,{timing},Key,,0,0,0,,{{\\an7\\pos({bx},{by})\\fad(70,180)\\p1\\bord2\\3c&H4DCCFF&\\c&H35414D&}}{rounded_box(104, 46)}'
            )
            lines.append(
                f'Dialogue: 4,{timing},Key,,0,0,0,,{{\\pos({bx + 52},{by + 23})\\fs28\\fad(70,180)}}Ctrl'
            )
    (directory / 'input-overlay.ass').write_text(header + '\n'.join(lines) + '\n')
    destination = directory / (output_name or f'review-{report["recording_fps"]}fps.mp4')
    subprocess.run(
        [
            'ffmpeg',
            '-nostdin',
            '-y',
            '-v',
            'warning',
            '-i',
            'walkthrough.mp4',
            '-vf',
            'pad=iw:ih+120:0:120:color=0x14171c,ass=input-overlay.ass',
            '-an',
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
            '-movflags',
            '+faststart',
            str(destination),
        ],
        cwd=directory,
        check=True,
    )
    print(destination)
    print(f'{destination.stat().st_size:,} bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output-name')
    args = parser.parse_args()
    render(args.directory, args.output_name)
