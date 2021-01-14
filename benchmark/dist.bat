python ../plot.py -f %* -x n_eval -y dist_mean --yerror dist_stddev --ylabel 遺伝子の値，標準偏差 --hide_legend
if errorlevel 1 pause
