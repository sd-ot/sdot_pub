N=`ls -a .*.files`
echo Makefile > $N
echo TODO.md >> $N
# echo TODO.txt >> $N
for d in lib src tests scripts benchmarks examples
do
    for t in '*.h' '*.cpp' '*.tcc' '*.cu' '*.txt' '*.py' '*.js' '*.html' '*.css' '*.files' '*.met' '*.coffee' '*.asm'
    do
        for i in `find $d -name "$t" -a -not -wholename "*/compilations/*"`
        do
            echo $i >> $N
            echo $i
            # git add $i
        done
    done
done
