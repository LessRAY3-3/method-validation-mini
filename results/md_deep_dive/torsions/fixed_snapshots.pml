reinitialize
bg_color white
set ray_opaque_background, 1
set cartoon_fancy_helices, 1
load /Users/lessray/Desktop/kimi code/method-validation-mini/results/md_deep_dive/torsions/seed_20261110_1500ps.pdb, s20261110_1500ps
color cyan, s20261110_1500ps
load /Users/lessray/Desktop/kimi code/method-validation-mini/results/md_deep_dive/torsions/seed_20261110_3500ps.pdb, s20261110_3500ps
color cyan, s20261110_3500ps
load /Users/lessray/Desktop/kimi code/method-validation-mini/results/md_deep_dive/torsions/seed_20261111_1500ps.pdb, s20261111_1500ps
color cyan, s20261111_1500ps
load /Users/lessray/Desktop/kimi code/method-validation-mini/results/md_deep_dive/torsions/seed_20261111_3500ps.pdb, s20261111_3500ps
color cyan, s20261111_3500ps
load /Users/lessray/Desktop/kimi code/method-validation-mini/results/md_deep_dive/torsions/seed_20261112_1500ps.pdb, s20261112_1500ps
color cyan, s20261112_1500ps
load /Users/lessray/Desktop/kimi code/method-validation-mini/results/md_deep_dive/torsions/seed_20261112_3500ps.pdb, s20261112_3500ps
color cyan, s20261112_3500ps
load /Users/lessray/Desktop/kimi code/method-validation-mini/data/prepared/1L2Y_model1.pdb, reference
color gray70, reference
hide everything
show cartoon
show sticks, resi 3+19 and not elem H
set stick_radius, 0.13
orient reference
zoom all, 2
disable all
enable reference
enable s20261110_1500ps
png /Users/lessray/Desktop/kimi code/method-validation-mini/results/md_deep_dive/torsions/snapshot_20261110_1500.png, 600, 450, ray=1
disable all
enable reference
enable s20261110_3500ps
png /Users/lessray/Desktop/kimi code/method-validation-mini/results/md_deep_dive/torsions/snapshot_20261110_3500.png, 600, 450, ray=1
disable all
enable reference
enable s20261111_1500ps
png /Users/lessray/Desktop/kimi code/method-validation-mini/results/md_deep_dive/torsions/snapshot_20261111_1500.png, 600, 450, ray=1
disable all
enable reference
enable s20261111_3500ps
png /Users/lessray/Desktop/kimi code/method-validation-mini/results/md_deep_dive/torsions/snapshot_20261111_3500.png, 600, 450, ray=1
disable all
enable reference
enable s20261112_1500ps
png /Users/lessray/Desktop/kimi code/method-validation-mini/results/md_deep_dive/torsions/snapshot_20261112_1500.png, 600, 450, ray=1
disable all
enable reference
enable s20261112_3500ps
png /Users/lessray/Desktop/kimi code/method-validation-mini/results/md_deep_dive/torsions/snapshot_20261112_3500.png, 600, 450, ray=1
quit
