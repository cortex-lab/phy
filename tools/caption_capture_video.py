"""Add in-window keycaps to a native Phy walkthrough recording.

The optional review cut concatenates selected original-time ranges. It never
speeds up the recording; the sidecar timeline records every retained range and
the short saved-session card that replaces the reopen wait.
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


def rounded_box(width, height, radius=9):
    return (
        f'm {radius} 0 l {width - radius} 0 '
        f'b {width} 0 {width} 0 {width} {radius} l {width} {height - radius} '
        f'b {width} {height} {width} {height} {width - radius} {height} '
        f'l {radius} {height} b 0 {height} 0 {height} 0 {height - radius} '
        f'l 0 {radius} b 0 0 0 0 {radius} 0'
    )


def _key_names(label):
    lowered = label.lower()
    if 'click' in lowered:
        keys = ['RIGHT CLICK' if label.startswith('Right click') else 'LEFT CLICK']
        return ['CTRL'] * label.startswith('Ctrl + ') + keys
    if ':' in label and label.split(':', 1)[0] in (
        'G',
        'Ctrl + Z',
        'Ctrl + Shift + Z',
        'Ctrl + S',
        'Enter',
        'Esc',
    ):
        return [part.upper() for part in label.split(':', 1)[0].split(' + ')]
    return []


def _press_release(event):
    label = event['label']
    if 'click' in label.lower():
        pressed = float(event.get('click_seconds', float(event['seconds']) + 0.9))
        return pressed, pressed + 0.25
    if ':' in label and label.split(':', 1)[0] in (
        'G',
        'Ctrl + Z',
        'Ctrl + Shift + Z',
        'Ctrl + S',
        'Enter',
        'Esc',
    ):
        pressed = float(event.get('key_seconds', float(event['seconds']) + 0.45))
        return pressed, float(event.get('key_released_seconds', pressed + 0.1))
    return None


def _ass_header(width, height):
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Key,DejaVu Sans,23,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1
Style: Card,DejaVu Sans,27,&H00FFFFFF,&H00FFFFFF,&H00000000,&H4017191D,0,0,0,0,100,100,0,0,3,2,0,5,36,36,18,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _keycap_lines(keys, start, end, width, height, y):
    if not keys:
        return []
    cap_widths = {
        'CTRL': 82,
        'SHIFT': 90,
        'LEFT CLICK': 148,
        'RIGHT CLICK': 166,
        'ENTER': 86,
        'G': 52,
        'Z': 52,
        'S': 52,
    }
    gap = 8
    cap_height = 48
    widths = [cap_widths.get(key, max(60, 16 + len(key) * 16)) for key in keys]
    total = sum(widths) + gap * (len(widths) - 1)
    x = (width - total) // 2
    start_ts, end_ts = timestamp(start), timestamp(end)
    lines = []
    for key, cap_width in zip(keys, widths):
        shape = rounded_box(cap_width, cap_height)
        lines.append(
            f'Dialogue: 1,{start_ts},{end_ts},Key,,0,0,0,,{{\\an7\\pos({x},{y})\\fad(90,160)\\p1\\bord2\\3c&HCCFF4D&\\1c&H17191D&\\1a&H40&}}{shape}'
        )
        lines.append(
            f'Dialogue: 2,{start_ts},{end_ts},Key,,0,0,0,,{{\\an5\\pos({x + cap_width // 2},{y + cap_height // 2})\\fs23\\bold1\\fad(90,160)\\1c&HFFFFFF&\\1a&H00&\\3a&H00&}}{key}'
        )
        x += cap_width + gap
    return lines


def _pointer_badge(event, start, end, width, height):
    if not event['label'].startswith('Ctrl + ') or 'point' not in event:
        return []
    px, py = event['point']
    badge_width, badge_height = 78, 42
    x = min(max(int(px) + 34, 4), width - badge_width - 4)
    y = min(max(int(py) - badge_height - 8, 4), height - badge_height - 4)
    start_ts, end_ts = timestamp(start), timestamp(end)
    return [
        f'Dialogue: 3,{start_ts},{end_ts},Key,,0,0,0,,{{\\an7\\pos({x},{y})\\fad(90,160)\\p1\\bord2\\3c&HCCFF4D&\\1c&H17191D&\\1a&H40&}}{rounded_box(badge_width, badge_height)}',
        f'Dialogue: 4,{start_ts},{end_ts},Key,,0,0,0,,{{\\an5\\pos({x + badge_width // 2},{y + badge_height // 2})\\fs20\\bold1\\fad(90,160)\\1c&HFFFFFF&\\1a&H00&}}CTRL',
    ]


def _review_items(events, duration, fps, target_seconds, checkpoints, reopen_card_seconds=0.8):
    """Select source-time clips and one cloned-frame reload card.

    The target is met by varying only the visible post-input settling dwell.
    Every source clip retains its original frames and rate.
    """
    reopen_index = next(
        (i for i, event in enumerate(events) if event['label'].startswith('Reopen')),
        None,
    )
    if reopen_index is None:
        raise ValueError('Review cuts require a Reopen saved working copy event.')

    reopen_seconds = float(events[reopen_index]['seconds'])

    final_checkpoint = max((float(step['seconds']) for step in checkpoints), default=duration)
    pre_reopen_checkpoint = max(
        (
            float(step['seconds'])
            for step in checkpoints
            if float(step['seconds']) <= reopen_seconds
        ),
        default=reopen_seconds,
    )

    def make_segments(tail):
        raw = []
        for index, event in enumerate(events):
            if index == reopen_index:
                continue
            label = event['label']
            start = max(0.0, float(event['seconds']) - 0.2)
            trigger = _press_release(event)
            if trigger:
                action_end = trigger[1]
            elif label.startswith(('View menu', 'Type evidence')):
                action_end = float(event['seconds']) + 0.35
            else:
                action_end = float(event['seconds']) + 0.4
            end = min(duration, action_end + tail)
            if index == len(events) - 1:
                end = max(end, min(duration, final_checkpoint + 0.3))
            if float(event['seconds']) < reopen_seconds:
                end = min(end, max(start + 1 / fps, reopen_seconds - 1 / fps))
            raw.append(
                {
                    'source_start': start,
                    'source_end': end,
                    'event_indices': [index],
                }
            )
        for step in checkpoints:
            step_time = float(step['seconds'])
            if step_time >= reopen_seconds - 1.1 / fps:
                continue  # The reopen card or post-reopen source represents this step.
            raw.append(
                {
                    'source_start': max(0.0, step_time - 0.3),
                    'source_end': min(duration, step_time + 0.3),
                    'event_indices': [],
                }
            )
        raw.sort(key=lambda item: item['source_start'])
        merged = []
        for segment in raw:
            if merged and segment['source_start'] <= merged[-1]['source_end'] + 0.12:
                merged[-1]['source_end'] = max(merged[-1]['source_end'], segment['source_end'])
                merged[-1]['event_indices'].extend(segment['event_indices'])
            else:
                merged.append(segment)
        return merged

    def make_items(tail):
        segments = make_segments(tail)
        items = []
        for segment in segments:
            first = max(0, round(segment['source_start'] * fps))
            last = max(first + 1, round(segment['source_end'] * fps))
            items.append(
                {
                    'kind': 'source',
                    'source_start_frame': first,
                    'source_end_frame': last,
                    'source_start': first / fps,
                    'source_end': last / fps,
                    'event_indices': segment['event_indices'],
                }
            )
        reopen_event = events[reopen_index]
        reopen_frame = max(
            0,
            round(float(reopen_event['seconds']) * fps) - 1,
            round(pre_reopen_checkpoint * fps),
        )
        reopen_frame = min(reopen_frame, max(0, int(reopen_seconds * fps)))
        card = {
            'kind': 'reload_card',
            'source_frame': reopen_frame,
            'source_time': reopen_frame / fps,
            'duration_frames': max(1, round(reopen_card_seconds * fps)),
            'duration': max(1, round(reopen_card_seconds * fps)) / fps,
            'event_indices': [reopen_index],
            'label': 'REOPENING SAVED SESSION — reload wait omitted',
        }
        items.append(card)
        items.sort(key=lambda item: item.get('source_start', item.get('source_time', 0.0)))
        return items

    def duration_for(tail):
        return sum(
            (item['source_end_frame'] - item['source_start_frame']) / fps
            if item['kind'] == 'source'
            else item['duration']
            for item in make_items(tail)
        )

    low, high = 0.0, duration
    minimum_duration = duration_for(low)
    if minimum_duration > target_seconds + 1 / fps:
        raise ValueError(
            f'Review cut cannot reach {target_seconds:.1f}s without trimming input actions '
            f'or final checkpoints (minimum is {minimum_duration:.2f}s).'
        )
    if duration_for(high) < target_seconds - 1 / fps:
        raise ValueError(
            f'Review cut cannot reach {target_seconds:.1f}s while preserving action timing.'
        )
    for _ in range(48):
        middle = (low + high) / 2
        if duration_for(middle) < target_seconds:
            low = middle
        else:
            high = middle
    tail = (low + high) / 2
    items = make_items(tail)

    output_cursor = 0.0
    event_output_times = {}
    for item in items:
        item['output_start'] = output_cursor
        if item['kind'] == 'source':
            item['output_end'] = (
                output_cursor + (item['source_end_frame'] - item['source_start_frame']) / fps
            )
            for index in item['event_indices']:
                event_output_times[index] = output_cursor + max(
                    0.0, float(events[index]['seconds']) - item['source_start']
                )
        else:
            item['output_end'] = output_cursor + item['duration']
            event_output_times[item['event_indices'][0]] = output_cursor
        output_cursor = item['output_end']
    checkpoint_output_times = {}
    for step in checkpoints:
        step_time = float(step['seconds'])
        item = next(
            (
                candidate
                for candidate in items
                if candidate['kind'] == 'source'
                and candidate['source_start'] <= step_time <= candidate['source_end']
            ),
            None,
        )
        if item is None:
            item = next(
                (
                    candidate
                    for candidate in items
                    if candidate['kind'] == 'reload_card'
                    and abs(step_time - candidate['source_time']) <= 1.1 / fps
                ),
                None,
            )
            checkpoint_output_times[step.get('name')] = item['output_start'] if item else None
        else:
            checkpoint_output_times[step.get('name')] = item['output_start'] + max(
                0.0, step_time - item['source_start']
            )
    return items, event_output_times, output_cursor, tail, checkpoint_output_times


def _filter_escape(path):
    return (
        str(path).replace('\\', '\\\\').replace(':', '\\:').replace("'", "\\'").replace(',', '\\,')
    )


def _omitted_ranges(items, duration):
    spans = sorted(
        (
            item['source_start'],
            item['source_end'],
        )
        for item in items
        if item['kind'] == 'source'
    )
    spans.extend(
        (item['source_time'], item['source_time'] + item['duration'])
        for item in items
        if item['kind'] != 'source'
    )
    spans.sort()
    omitted = []
    cursor = 0.0
    card_time = next(
        (item['source_time'] for item in items if item['kind'] == 'reload_card'), None
    )
    for start, end in spans:
        if start > cursor + 1 / 60:
            reopen_gap = card_time is not None and cursor <= card_time <= start
            omitted.append(
                {
                    'source_start': cursor,
                    'source_end': start,
                    'duration_seconds': start - cursor,
                    'reason': 'reopen wait replaced by reload card'
                    if reopen_gap
                    else 'idle or settled dwell trimmed',
                }
            )
        cursor = max(cursor, end)
    if cursor < duration - 1 / 60:
        omitted.append(
            {
                'source_start': cursor,
                'source_end': duration,
                'duration_seconds': duration - cursor,
                'reason': 'trailing settled dwell trimmed',
            }
        )
    return omitted


def render(directory, output_name=None, review_cut=False, review_seconds=43.0, overwrite=False):
    directory = directory.resolve()
    report = json.loads((directory / 'capture.json').read_text())
    if not report['success'] or not report['source_unchanged']:
        raise ValueError('Capture must report success and unchanged source data.')
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
    width, height = int(video['width']), int(video['height'])
    duration = float(details['format']['duration'])
    fps_num, fps_den = map(int, video['r_frame_rate'].split('/'))
    fps = fps_num / fps_den
    events = report['input_events']
    if review_cut and not 40.0 <= review_seconds <= 45.0:
        raise ValueError('--review-seconds must be between 40 and 45 seconds.')

    items = None
    event_output_times = {i: float(event['seconds']) for i, event in enumerate(events)}
    review_duration = duration
    tail = None
    if review_cut:
        items, event_output_times, review_duration, tail, checkpoint_output_times = _review_items(
            events, duration, fps, review_seconds, report.get('steps', [])
        )

    output_name = output_name or (
        'review-integrated.mp4' if review_cut else f'review-{report["recording_fps"]}fps.mp4'
    )
    destination = Path(output_name)
    if not destination.is_absolute():
        destination = directory / destination
    destination = destination.resolve()
    if destination == (directory / 'walkthrough.mp4').resolve():
        raise ValueError('Output must not overwrite the original walkthrough.')
    if destination.exists() and not overwrite:
        raise FileExistsError(f'Refusing to overwrite existing output: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    ass_path = destination.with_suffix('.ass')
    timeline_path = destination.with_suffix('.timeline.json')
    if not overwrite and (ass_path.exists() or (review_cut and timeline_path.exists())):
        raise FileExistsError('Refusing to overwrite an existing caption or timeline sidecar.')

    lines = []
    key_y = max(0, height - 140)
    for index, event in enumerate(events):
        keys = _key_names(event['label'])
        if not keys:
            continue
        trigger = _press_release(event)
        if trigger is None:
            continue
        pressed, released = trigger
        output_event_time = event_output_times[index]
        input_event_time = float(event['seconds'])
        offset = output_event_time - input_event_time
        cue_start = max(0.0, pressed - 0.15 + offset)
        cue_end = released + 0.85 + offset
        if review_cut:
            containing = next(item for item in items if index in item.get('event_indices', []))
            cue_end = min(cue_end, containing['output_end'])
        lines.extend(_keycap_lines(keys, cue_start, cue_end, width, height, key_y))
        lines.extend(_pointer_badge(event, cue_start, cue_end, width, height))

    if review_cut:
        card = next(item for item in items if item['kind'] == 'reload_card')
        card_start, card_end = timestamp(card['output_start']), timestamp(card['output_end'])
        card_width, card_height = min(650, width - 80), 112
        x, y = (width - card_width) // 2, (height - card_height) // 2
        lines.append(
            f'Dialogue: 1,{card_start},{card_end},Key,,0,0,0,,{{\\an7\\pos({x},{y})\\fad(80,130)\\p1\\bord2\\3c&HCCFF4D&\\1c&H17191D&\\1a&H40&}}{rounded_box(card_width, card_height, 14)}'
        )
        lines.append(
            f'Dialogue: 2,{card_start},{card_end},Key,,0,0,0,,{{\\an5\\pos({width // 2},{height // 2 - 14})\\fs28\\bold1\\fad(80,130)\\1c&HFFFFFF&\\1a&H00&}}REOPENING SAVED SESSION'
        )
        lines.append(
            f'Dialogue: 2,{card_start},{card_end},Key,,0,0,0,,{{\\an5\\pos({width // 2},{height // 2 + 22})\\fs20\\fad(80,130)\\1c&HFFFFFF&\\1a&H00&}}Reload wait omitted'
        )

    ass_path.write_text(_ass_header(width, height) + '\n'.join(lines) + '\n')
    command = [
        'ffmpeg',
        '-nostdin',
        '-y' if overwrite else '-n',
        '-v',
        'warning',
        '-i',
        str(directory / 'walkthrough.mp4'),
    ]
    if review_cut:
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
                filters.append(
                    f'[0:v]trim=start_frame={frame}:end_frame={frame + 1},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={max(0.0, item["duration"] - 1 / fps):.6f}[{label}]'
                )
        ass_filter = _filter_escape(ass_path)
        filters.append(''.join(labels) + f'concat=n={len(labels)}:v=1:a=0,ass={ass_filter}[outv]')
        command.extend(
            [
                '-filter_complex',
                ';'.join(filters),
                '-map',
                '[outv]',
            ]
        )
    else:
        command.extend(['-vf', f'ass={_filter_escape(ass_path)}'])
    command.extend(
        [
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
            '-r',
            f'{fps_num}/{fps_den}',
            '-fps_mode',
            'cfr',
            '-movflags',
            '+faststart',
            str(destination),
        ]
    )
    subprocess.run(command, cwd=directory, check=True)

    if review_cut:
        timeline = {
            'source': str(directory / 'walkthrough.mp4'),
            'output': str(destination),
            'source_fps': f'{fps_num}/{fps_den}',
            'source_duration_seconds': duration,
            'output_duration_seconds': review_duration,
            'review_cut': True,
            'timeline_note': 'Original-speed source frames concatenated; idle dwell and reopen wait omitted. Not a latency benchmark.',
            'settle_tail_seconds': tail,
            'items': items,
            'omitted_source_ranges': _omitted_ranges(items, duration),
            'events': [
                {
                    'index': index,
                    'label': event['label'],
                    'source_seconds': float(event['seconds']),
                    'output_seconds': event_output_times[index],
                }
                for index, event in enumerate(events)
            ],
            'checkpoints': [
                {
                    'name': step.get('name'),
                    'source_seconds': float(step['seconds']),
                    'output_seconds': checkpoint_output_times.get(step.get('name')),
                    'captured': checkpoint_output_times.get(step.get('name')) is not None,
                }
                for step in report.get('steps', [])
            ],
        }
        timeline_path.write_text(json.dumps(timeline, indent=2) + '\n')
        print(timeline_path)
    print(destination)
    print(f'{destination.stat().st_size:,} bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output-name')
    parser.add_argument('--review-cut', action='store_true')
    parser.add_argument('--review-seconds', type=float)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    render(
        args.directory,
        args.output_name,
        review_cut=args.review_cut or args.review_seconds is not None,
        review_seconds=args.review_seconds if args.review_seconds is not None else 43.0,
        overwrite=args.overwrite,
    )
