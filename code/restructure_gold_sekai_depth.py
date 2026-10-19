"""One-off reorganizer: /lp-dev/jmurrugarral/EVP/depth/gold_sekai_900_depth_res/ (900 clip dirs,
clip_XXXX/00001.png..00050.png, flat) -> EgoGroups_test/videos_frames-style layout
(clip_XXXX_YYYYY/00001.png..00010.png, 5 sub-clips of 10 frames each per base clip), matching
EgoGroups_test's videos_frames convention (frame_ids 1-10 per sub-clip, sub-clip index YYYYY =
00000-00004, frames [sub_clip*10+1 .. sub_clip*10+10] of the base clip renumbered locally 1-10).

Source frames are untouched (copied, not moved); output uses hardlinks when possible (same
filesystem) to avoid duplicating ~45,000 images on disk, falling back to a real copy otherwise.
Idempotent/resumable: skips an output sub-clip dir that already has all 10 frames.
"""

import os
import shutil
import sys

SRC_DIR = '/lp-dev/jmurrugarral/EVP/depth/gold_sekai_900_depth_res'
DST_DIR = '/lp-dev/jmurrugarral/EVP/depth/gold_sekai_900_depth_res_egogroups_style'
TOTAL_CLIPS = 900
SUB_CLIPS = 5
FRAMES_PER_SUB_CLIP = 10
EXT = '.png'


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    done, filled, skipped_missing = 0, 0, 0

    for clip_num in range(1, TOTAL_CLIPS + 1):
        clip_name = f'clip_{clip_num:04d}'
        src_clip_dir = os.path.join(SRC_DIR, clip_name)
        if not os.path.isdir(src_clip_dir):
            print(f'WARNING: missing source clip dir {src_clip_dir}')
            skipped_missing += 1
            continue

        for sub_clip in range(SUB_CLIPS):
            sub_clip_name = f'{clip_name}_{sub_clip:05d}'
            dst_sub_dir = os.path.join(DST_DIR, sub_clip_name)

            if os.path.isdir(dst_sub_dir) and \
                    len(os.listdir(dst_sub_dir)) == FRAMES_PER_SUB_CLIP:
                done += 1
                continue

            os.makedirs(dst_sub_dir, exist_ok=True)
            base_frame_start = sub_clip * FRAMES_PER_SUB_CLIP + 1  # 1,11,21,31,41

            ok = True
            for local_idx in range(1, FRAMES_PER_SUB_CLIP + 1):
                src_frame_num = base_frame_start + local_idx - 1
                src_path = os.path.join(src_clip_dir, f'{src_frame_num:05d}{EXT}')
                dst_path = os.path.join(dst_sub_dir, f'{local_idx:05d}{EXT}')
                if not os.path.exists(src_path):
                    print(f'WARNING: missing source frame {src_path}')
                    ok = False
                    continue
                if not os.path.exists(dst_path):
                    link_or_copy(src_path, dst_path)

            if ok:
                filled += 1
            else:
                skipped_missing += 1

    print(f'Done. Already-complete sub-clips: {done}, newly filled: {filled}, '
          f'sub-clips/clips with missing frames: {skipped_missing}')
    print(f'Output: {DST_DIR}')


if __name__ == '__main__':
    sys.exit(main())
