BEST_DEV=`grep Dev $1|sort -k6|tail -1`
echo $BEST_DEV
BEST_TEST=`echo $BEST_DEV| sed 's/Development/Test/g'| awk '{$6=""; print}'`
grep "$BEST_TEST" $1

