"""
RUN THIS ALONE FIRST. It cannot crash — it just lists what's really
there. Paste the FULL output back before we touch the real pipeline
again. This ends the guess-a-path-and-fail cycle permanently: every
path in the next script will be copied directly from this output,
not guessed from old notebook logs.
"""
import os

print("="*70)
print("WHAT IS ACTUALLY MOUNTED UNDER /kaggle/input RIGHT NOW")
print("="*70)

if not os.path.exists('/kaggle/input'):
    print("!! /kaggle/input does not exist in this environment at all.")
else:
    for root, dirs, files in os.walk('/kaggle/input'):
        depth = root.count(os.sep) - '/kaggle/input'.count(os.sep)
        if depth > 3:
            dirs[:] = []
            continue
        indent = '  ' * depth
        print(f"{indent}{os.path.basename(root) or root}/")
        for f in files:
            full = os.path.join(root, f)
            try:
                size_mb = os.path.getsize(full) / (1024*1024)
                print(f"{indent}  {f}  ({size_mb:.1f} MB)")
            except Exception:
                print(f"{indent}  {f}")

print()
print("="*70)
print("ALL .csv FILES FOUND (full absolute paths - copy these exactly)")
print("="*70)
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f.lower().endswith('.csv'):
            print(os.path.join(root, f))
