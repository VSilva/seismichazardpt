import geopandas as gpd
import pandas as pd
import numpy as np
import os

!import os
import re
from pathlib import Path

!gmt set FONT_ANNOT_PRIMARY=11p
!gmt set PS_CHAR_ENCODING=Standard+
!gmt set FONT_LABEL=11p
!gmt set MAP_LABEL_OFFSET=6p

hazard_maps = ['../seismic_hazard/AbrahamsonEtAl2014/outputs/quantile_map-0.05_3.csv',
               # '../seismic_hazard/AbrahamsonEtAl2014/outputs/hazard_map-mean_3.csv',
               # '../seismic_hazard/AbrahamsonEtAl2014/outputs/quantile_map-0.95_3.csv',

               # '../seismic_hazard/BooreEtAl2014/outputs/quantile_map-0.05_4.csv',
               # '../seismic_hazard/BooreEtAl2014/outputs/hazard_map-mean_4.csv',
               # '../seismic_hazard/BooreEtAl2014/outputs/quantile_map-0.95_4.csv',

               # '../seismic_hazard/KothaEtAl2020_highstress/outputs/quantile_map-0.05_5.csv',
               # '../seismic_hazard/KothaEtAl2020_highstress/outputs/hazard_map-mean_5.csv',
               # '../seismic_hazard/KothaEtAl2020_highstress/outputs/quantile_map-0.95_5.csv',
               
               '../seismic_hazard/KothaEtAl2020_midstress/outputs/quantile_map-0.05_13.csv',
               '../seismic_hazard/KothaEtAl2020_midstress/outputs/hazard_map-mean_13.csv',
               '../seismic_hazard/KothaEtAl2020_midstress/outputs/quantile_map-0.95_13.csv',

               # '../seismic_hazard/MorgadoEtAl/outputs/quantile_map-0.05_8.csv',
               # '../seismic_hazard/MorgadoEtAl/outputs/hazard_map-mean_8.csv',
               # '../seismic_hazard/MorgadoEtAl/outputs/quantile_map-0.95_8.csv',

               # '../seismic_hazard/TaherianEtAl_modified/outputs/quantile_map-0.05_1.csv',
               # '../seismic_hazard/TaherianEtAl_modified/outputs/hazard_map-mean_1.csv',
               # '../seismic_hazard/TaherianEtAl_modified/outputs/quantile_map-0.95_1.csv'
               
               # '../seismic_hazard/GMMLogicTree/outputs/quantile_map-0.05_12.csv',
               # '../seismic_hazard/GMMLogicTree/outputs/hazard_map-mean_12.csv',
               '../seismic_hazard/GMMLogicTree/outputs/quantile_map-0.95_12.csv']

IMs = ['PGA','SA(1.0)']
rates = ['0.002105','0.000404']
cpts = ['./cpts/hazard_efehr_max0p3.cpt','./cpts/hazard_efehr_max0p4.cpt']

for hazard_csv in hazard_maps:
    print(hazard_csv)

    # --- Build title from: folder after "seismic_hazard" + csv stem (optionally dropping trailing _###) ---
    p = Path(hazard_csv)
    parts = p.parts
    i = parts.index("seismic_hazard")
    model_name = parts[i + 1]                       # e.g., KothaEtAl2020_midstress

    csv_stem = p.stem                               # e.g., quantile_map-0.05_181
    csv_stem = re.sub(r"_\d+$", "", csv_stem)       # -> quantile_map-0.05  (remove this line if you want to keep _181)

    hazard_title = f"{model_name}_{csv_stem}"       # e.g., KothaEtAl2020_midstress_quantile_map-0.05

    # --- Region (Portugal + buffer) and projection ---
    R = "-R-10.1/-5.5/36.8/42.3"
    J = "-JT-8/18c"

    # --- Build a smooth hazard grid from irregular CSV points (lon,lat,mean) ---
    Ihaz = "0.01"
    
    for i in range(len(IMs)):
        for j in range(len(rates)):
            
            safe_im = (IMs[i].replace("SA(", "SA").replace(")", "s").replace(".", "p").replace("(", ""))  # e.g. "SA(1.0)" -> "SA1p0s"
            ps_out  = f'./maps/{hazard_title}_{safe_im}_{rates[j]}.ps'
            png_out = f'./maps/{hazard_title}_{safe_im}_{rates[j]}.png'

            column_name = IMs[i]+'-'+rates[j]
            
            with open(hazard_csv, "r", encoding="utf-8") as f:
                header_lines = [next(f).strip() for _ in range(2)]
            header = re.split(r"\s*,\s*", header_lines[-1].lstrip("#"))
            icol = header.index(column_name)
                
            !/opt/homebrew/bin/gmt convert {hazard_csv} -hi2 -i0,1,{icol} > ./hazard/hazard.xyz            
            !/opt/homebrew/bin/gmt surface ./hazard/hazard.xyz {R} -I{Ihaz} -G./grds/hazard.grd=bf -T0.35 -V

    # --- Mask hazard to Portugal only ---
            !/opt/homebrew/bin/gmt pscoast {R} -Df -M -EPT > ./gmts/portugal.gmt
            !/opt/homebrew/bin/gmt grdmask ./gmts/portugal.gmt {R} -I{Ihaz} -G./grds/pt_mask.grd=bf -NNaN/1/1 -V
            !/opt/homebrew/bin/gmt grdmath ./grds/hazard.grd=bf ./grds/pt_mask.grd=bf MUL = ./grds/hazard_pt.grd=bf

    # --- Base map ---
            !/opt/homebrew/bin/gmt pscoast {R} {J} -N1 -Bf1a1 -Df -A100+l -K -S200/230/255 -G220 -Wthinnest > {ps_out}

    # --- Hazard fill (NO shading) ---
            !/opt/homebrew/bin/gmt grdimage ./grds/hazard_pt.grd=bf {R} {J} -O -K -C{cpts[j]} -V -nn -Q >> {ps_out}

    # --- Contours ---
            !/opt/homebrew/bin/gmt grdcontour ./grds/hazard_pt.grd=bf {R} {J} -O -K -C0.05 -A0.05+f14p,Helvetica,black -Gd5c -W0.6p,black >> {ps_out}

    # --- Coastlines / borders on top ---
            !/opt/homebrew/bin/gmt pscoast {R} {J} -O -K -Df -A100+l -Wthinnest -N1/0.5p >> {ps_out}

    # --- Map title (top center, using map title mechanism) ---
    # !/opt/homebrew/bin/gmt psbasemap {R} {J} -O -K -B+t"{hazard_title}" >> {ps_out}

    # --- Colorbar with legend text above it (attached to psscale) ---
    #!/opt/homebrew/bin/gmt psscale -Dx1c/1c+w10c/0.4c+h+ml -O -K -L -C./cpts/hazard_efehr_max0p3.cpt -Bx+l"PGA (g)" >> {ps_out}
            IM_text = IMs[i]+' (g)'
            !/opt/homebrew/bin/gmt psscale {R} {J} -O -K -DJMR+w16c/1.0c+v+o1.5c/0c -C{cpts[j]} -L -By+l'{IM_text}' >> {ps_out}


    # --- Cities ---
            !/opt/homebrew/bin/gmt pstext ./dat/cities.txt {R} {J} -O -K -F+jL+f10p,Helvetica,black=2p,white -D0.05c >> {ps_out}
            !/opt/homebrew/bin/gmt pstext ./dat/cities.txt {R} {J} -O -K -F+jL+f10p,Helvetica -D0.05c >> {ps_out}

    # --- Close PS and convert to PNG (CSV-based filename) ---
            !/opt/homebrew/bin/gmt psxy -T -O >> {ps_out}
            !convert -density 150 -trim {ps_out} -rotate 90 {png_out}



from pathlib import Path

def delete_ps_files(folder: str | Path, recursive: bool = False) -> int:
    folder = Path(folder)
    pattern = "**/*.ps" if recursive else "*.ps"

    deleted = 0
    for ps_file in folder.glob(pattern):
        if ps_file.is_file():
            ps_file.unlink()
            deleted += 1
    return deleted

# Example:
n = delete_ps_files("./maps", recursive=False)
print(f"Deleted {n} .ps files from ./maps")