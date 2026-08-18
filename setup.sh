#!/usr/bin/env bash
# One-time setup for the Reasoning_via_Inference project.
# Run from the repository root:  bash setup.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$ROOT/Report code scripts"

echo "==> Setting up symlinks in 'Report code scripts/' ..."
cd "$SCRIPTS"
for d in contact_graspnet_pytorch generated_scenes results assets mujoco_menagerie; do
  if [ -e "$d" ] && [ ! -L "$d" ]; then
    echo "    SKIP $d (exists and is not a symlink)"
  elif [ -L "$d" ]; then
    echo "    OK   $d (already linked)"
  else
    ln -s "../$d" "$d"
    echo "    LINK $d -> ../$d"
  fi
done

echo ""
echo "==> Installing Python dependencies ..."
pip install -r "$SCRIPTS/requirements.txt"

echo ""
echo "==> Sanity checks ..."
python3 -c "import mujoco, torch, pandas; print('  packages OK')"
python3 -c "
import mujoco, os
p = os.path.join('$ROOT', 'generated_scenes', 'scene_cylinder.xml')
m = mujoco.MjModel.from_xml_path(p)
print('  MuJoCo scene loads OK:', p)
"
python3 -c "
import os, sys
sys.path.insert(0, os.path.join('$ROOT', 'contact_graspnet_pytorch', 'contact_graspnet_pytorch'))
import torch
print('  CUDA available:', torch.cuda.is_available(), '(CPU is fine for demos)')
"

echo ""
echo "Setup complete."
echo ""
echo "Next steps (from 'Report code scripts/'):"
echo "  cd \"$SCRIPTS\""
echo "  mjpython demo_floating_gripper.py --object cylinder    # macOS interactive demo"
echo "  python3 visualize_cgn_grasps.py --save                 # grasp visualisation"
echo "  python3 score_algorithm2.py                            # re-score SCM diagnosis"
