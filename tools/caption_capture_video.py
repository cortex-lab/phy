"""Add an input-event footer to a private-display GUI recording.

The GUI picture is neither cropped nor overlaid: a separate 72px footer shows
recorded click/key labels. The output keeps the input frame rate and timing.
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


def render(directory):
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
PlayResY: {video['height'] + 72}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Input,DejaVu Sans,32,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,1,24,24,18,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for index, event in enumerate(events):
        start = float(event['seconds'])
        stop = float(events[index + 1]['seconds']) if index + 1 < len(events) else duration
        label = event['label'].replace('\n', ' ').replace('{', '').replace('}', '')
        lines.append(
            f'Dialogue: 0,{timestamp(start)},{timestamp(stop)},Input,,0,0,0,,DEMO INPUT   {label}'
        )
    (directory / 'input-overlay.ass').write_text(header + '\n'.join(lines) + '\n')
    destination = directory / f'review-{report["recording_fps"]}fps.mp4'
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
            'pad=iw:ih+72:0:0:color=0x14171c,ass=input-overlay.ass',
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
    render(parser.parse_args().directory)
