# One-off data-maintenance script: jsons_step2_new preserves the track_id numbering the EgoGroups
# ground truth (F1_evaluator/out/gt_gold_sekai_{2,22,42}.pkl) was actually annotated against, but
# jsons_step2/step3/step4/step5 (the pipeline stages that add depth/3D fields) silently renumber
# track_ids for the same underlying detections. Verified empirically (see conversation this script
# was written for): across a 3000-frame sample, detection count and bbox order are always
# identical between jsons_step2_new and jsons_step5 for the same clip/frame, only track_id
# differs, in 78.3% of frames -- matching almost exactly the 78.1% GT/detection person_id
# mismatch rate measured against jsons_step5. This script overwrites each detection's track_id in
# the later-stage files with jsons_step2_new's track_id at the same detection index, restoring
# GT alignment, while leaving every other field (bbox, depth, etc.) untouched.
#
# Safety:
#   - Defaults to a DRY RUN: reports what would change, writes nothing.
#   - Only actually writes with --apply.
#   - Before writing a file for the first time, saves an untouched copy alongside it as
#     '<file>.bak' (skipped if a .bak already exists, so re-running --apply after a first
#     successful run does not overwrite the backup with already-fixed data).
#   - Per frame, only remaps track_ids if jsons_step2_new and the target file agree exactly on
#     detection count AND bbox order for that frame; any frame that fails this check is left
#     untouched and logged, never guessed at.
#
# Usage:
#   python3 fix_egogroups_track_ids.py                      # dry run, all four stages
#   python3 fix_egogroups_track_ids.py --stages step5        # dry run, just jsons_step5
#   python3 fix_egogroups_track_ids.py --apply               # actually write the fix
#   python3 fix_egogroups_track_ids.py --apply --stages step2 step3 step4 step5

import argparse
import glob
import json
import os
import shutil

SEKAI_GOLD_DIR = '/lp-dev/jmurrugarral/gold_SEKAI_900_3'
SOURCE_DIR = os.path.join(SEKAI_GOLD_DIR, 'jsons_step2_new')
ALL_STAGES = ['step2', 'step3', 'step4', 'step5']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Overwrite track_id fields in jsons_step2/3/4/5 with jsons_step2_new\'s '
                    'track_ids (the ones the EgoGroups ground truth was annotated against), '
                    'matched positionally per frame after verifying bbox order agrees exactly.'
    )
    parser.add_argument(
        '--stages', nargs='+', choices=ALL_STAGES, default=ALL_STAGES,
        help='Which jsons_stepN directories to fix (default: all four).'
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Actually write changes. Without this flag, only reports what would change.'
    )
    return parser.parse_args()


def load_source_clip(clip_name):
    path = os.path.join(SOURCE_DIR, clip_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return {fr['frame_id']: fr['detections'] for fr in data['frames']}


def fix_clip(target_path, source_frames, apply, log_prefix):
    with open(target_path) as f:
        target_data = json.load(f)

    n_frames_total = len(target_data['frames'])
    n_frames_fixed = 0
    n_frames_unchanged_already_matching = 0
    n_frames_skipped = []
    n_detections_changed = 0
    changed = False

    for frame in target_data['frames']:
        fid = frame['frame_id']
        src_dets = source_frames.get(fid)
        if src_dets is None:
            n_frames_skipped.append((fid, 'no matching frame in jsons_step2_new'))
            continue

        tgt_dets = frame['detections']
        if len(src_dets) != len(tgt_dets):
            n_frames_skipped.append((fid, f'detection count differs (src={len(src_dets)}, tgt={len(tgt_dets)})'))
            continue

        bboxes_match = all(s['bbox'] == t['bbox'] for s, t in zip(src_dets, tgt_dets))
        if not bboxes_match:
            n_frames_skipped.append((fid, 'bbox order/values differ'))
            continue

        frame_changed = False
        for s_det, t_det in zip(src_dets, tgt_dets):
            if t_det['track_id'] != s_det['track_id']:
                t_det['track_id'] = s_det['track_id']
                n_detections_changed += 1
                frame_changed = True
        if frame_changed:
            n_frames_fixed += 1
            changed = True
        else:
            n_frames_unchanged_already_matching += 1

    if changed and apply:
        bak_path = target_path + '.bak'
        if not os.path.exists(bak_path):
            shutil.copy2(target_path, bak_path)
        with open(target_path, 'w') as f:
            json.dump(target_data, f)

    return {
        'n_frames_total': n_frames_total,
        'n_frames_fixed': n_frames_fixed,
        'n_frames_unchanged_already_matching': n_frames_unchanged_already_matching,
        'n_frames_skipped': n_frames_skipped,
        'n_detections_changed': n_detections_changed,
        'changed': changed,
    }


def main():
    args = parse_args()
    mode = 'APPLY' if args.apply else 'DRY RUN'
    print(f'mode: {mode}')
    print(f'stages: {args.stages}')
    print()

    for stage in args.stages:
        stage_dir = os.path.join(SEKAI_GOLD_DIR, f'jsons_{stage}')
        clip_paths = sorted(glob.glob(os.path.join(stage_dir, 'clip_*.json')))
        print(f'=== jsons_{stage}: {len(clip_paths)} clips ===')

        total_frames_fixed = 0
        total_detections_changed = 0
        total_frames_skipped = 0
        clips_with_skips = []
        clips_touched = 0

        for clip_path in clip_paths:
            clip_name = os.path.basename(clip_path)
            source_frames = load_source_clip(clip_name)
            if source_frames is None:
                print(f'  WARNING: no matching {clip_name} in jsons_step2_new, skipping entire clip')
                continue

            result = fix_clip(clip_path, source_frames, args.apply, log_prefix=f'jsons_{stage}/{clip_name}')
            total_frames_fixed += result['n_frames_fixed']
            total_detections_changed += result['n_detections_changed']
            total_frames_skipped += len(result['n_frames_skipped'])
            if result['n_frames_skipped']:
                clips_with_skips.append((clip_name, result['n_frames_skipped']))
            if result['changed']:
                clips_touched += 1

        print(f'  clips touched (had >=1 track_id fixed): {clips_touched}/{len(clip_paths)}')
        print(f'  frames fixed: {total_frames_fixed}')
        print(f'  individual detection track_ids changed: {total_detections_changed}')
        print(f'  frames skipped (count/bbox mismatch or no source frame): {total_frames_skipped}')
        if clips_with_skips:
            print(f'  clips with at least one skipped frame: {len(clips_with_skips)} (first 5 shown)')
            for clip_name, skips in clips_with_skips[:5]:
                for fid, reason in skips[:3]:
                    print(f'    {clip_name} frame {fid}: {reason}')
        print()

    if not args.apply:
        print('Dry run complete -- no files were modified. Re-run with --apply to write changes.')
    else:
        print('Applied. Original files backed up alongside as <file>.bak before any changes.')


if __name__ == '__main__':
    main()
