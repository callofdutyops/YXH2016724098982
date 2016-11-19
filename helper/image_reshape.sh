mkdir newdir
mogrify -path newdir -resize 96x96! *.png
rm -rf *.png
mv newdir/*.png .
rm -rf newdir
