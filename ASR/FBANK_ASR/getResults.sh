# Getting results [see RESULTS file]

base=$1
while read y
 do
    x="exp/${base}/test/${y}/decode/"
    y=`echo $x | xargs -n1 dirname`
    #echo "$y"
[ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh | awk -vvar=$y '{print var" "$2}'
 done < dirList

