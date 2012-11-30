set autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "BioHadoop"
set yrange [0 : 400]
set xlabel "Workers"
set ylabel "Time in seconds"
set grid x y
unset colorbox


set style line 1 lt 1 lw 1 
set style line 2 lt 2 lw 1 
set style line 3 lt 3 lw 1 

plot \
"plotdata.dat" using 1:2 title 'Average' with lines ls 2, \
"plotdata.dat" using 1:3:4:5 title 'Results with errorbars' with yerrorbars ls 1

set term png
set output "speedplot.png"

replot
                                                                                                       
