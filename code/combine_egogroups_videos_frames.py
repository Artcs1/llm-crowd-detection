"""Reconstruct EgoGroups/videos_frames from EgoGroups_test + EgoGroups_train's sub-clips.

EgoGroups_test/EgoGroups_train split each original 50-frame clip into 5 sub-clips of 10 frames
each (clip_XXXX_YYYYY). Confirmed via jsons_step1 that the 5 sub-clips are genuinely sequential,
non-overlapping segments (start_frame 1/11/21/31/41), not duplicate samples -- so this is a lossless
reconstruction: sub-clip i's frame j (1-10) becomes the combined clip's frame i*10+j (1-50).

Base clip numbering: 0001-0900 from EgoGroups_test, 0901-2700 from EgoGroups_train (continuing
EgoGroups_train's own existing numbering convention). Idempotent/resumable -- skips any base clip
whose combined directory already exists.
"""

import os
import shutil

from tqdm import tqdm

EGOTEST_DIR = '/lp-dev/jmurrugarral/EgoGroups_test/videos_frames'
EGOTRAIN_DIR = '/lp-dev/jmurrugarral/EgoGroups_train/videos_frames'
OUT_DIR = '/lp-dev/jmurrugarral/EgoGroups/videos_frames'

TOTAL_CLIPS = 2700
SUB_CLIPS = 5
FRAMES_PER_SUB_CLIP = 10


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    missing = [n for n in range(1, TOTAL_CLIPS + 1)
               if not os.path.isdir(os.path.join(OUT_DIR, f'clip_{n:04d}'))]
    print(f'{TOTAL_CLIPS - len(missing)}/{TOTAL_CLIPS} clips already exist; reconstructing {len(missing)} missing clips')

    for n in tqdm(missing):
        src_root = EGOTEST_DIR if n <= 900 else EGOTRAIN_DIR

        sub_clip_dirs = [os.path.join(src_root, f'clip_{n:04d}_{i:05d}') for i in range(SUB_CLIPS)]
        if not all(os.path.isdir(d) and len(os.listdir(d)) == FRAMES_PER_SUB_CLIP for d in sub_clip_dirs):
            print(f'WARNING: clip_{n:04d} has incomplete source sub-clips, skipping')
            continue

        out_dir = os.path.join(OUT_DIR, f'clip_{n:04d}')
        os.makedirs(out_dir, exist_ok=True)

        for i, sub_clip_dir in enumerate(sub_clip_dirs):
            for j in range(1, FRAMES_PER_SUB_CLIP + 1):
                src = os.path.join(sub_clip_dir, f'{j:05d}.jpeg')
                dst = os.path.join(out_dir, f'{i * FRAMES_PER_SUB_CLIP + j:05d}.jpeg')
                shutil.copy2(src, dst)


if __name__ == '__main__':
    main()
