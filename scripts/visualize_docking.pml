load data/raw/1iep_receptorH.pdb, receptor
load data/raw/1iep_ligand.sdf, crystal
load results/docking/vina_e32_s42.sdf, docked
load data/prepared/1iep_receptor.box.pdb, box
bg_color white
set ray_opaque_background, on
hide everything
show cartoon, receptor
color gray80, receptor
show sticks, crystal
color magenta, crystal
show sticks, docked and state 1
color cyan, docked and state 1
show lines, box
color black, box
set stick_radius, 0.16
set cartoon_transparency, 0.25
zoom crystal or (docked and state 1), 8
orient crystal or (docked and state 1)
disable box
png figures/docking_overlay.png, width=1400, height=1000, dpi=200, ray=1
enable box
disable docked
zoom box, 4
png figures/docking_box.png, width=1400, height=1000, dpi=200, ray=1
