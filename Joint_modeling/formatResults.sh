
base=$1
echo $base
res=`sh getResults.sh $base | awk '{print $NF}' |  tr -s '\n' ' '`
echo $res
echo $res | awk '{print "Cond Sim_dt", ($1+$2+$3+$4+$5+$6)/6}' 
echo $res | awk '{print "Cond Sim_et", ($13+$14+$9+$10+$11+$12)/6}' 
echo $res | awk '{print "Cond Real_dt", ($7+$8)/2}' 
echo $res | awk '{print "Cond Real_et", ($15+$16)/2}' 
