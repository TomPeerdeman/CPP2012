set autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "Speedup compared to sequential"
set yrange [0 : 4]
set xlabel "Workers"
set ylabel "Speedup"
set grid x y
unset colorbox


set style line 1 lt 1 lw 1 
set style line 2 lt 2 lw 1 
set style line 3 lt 3 lw 1 

plot \
"speedplotdata.dat" using 1:2 title 'Sequential' with lines ls 2, \
"speedplotdata.dat" using 1:3 title 'Speed' with lines ls 1

set term png
set output "speedup.png"

replot
                                                                                                       
