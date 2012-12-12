set autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title ""
set xlabel "Numbers in list"
set ylabel "Time in microseconds"
set grid x y
unset colorbox


set style line 1 lt 1 lw 1 
set style line 2 lt 2 lw 1 
set style line 3 lt 3 lw 1 

plot \
"plotdata.dat" using 1:3 title 'GPU' with lines ls 1,\
"plotdata.dat" using 1:4 title 'GPU optimized' with lines ls 2

set term png
set output "6_2_2.png"

replot
                                                                                                       
