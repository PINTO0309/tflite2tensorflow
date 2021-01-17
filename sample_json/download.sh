#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1_3-2by5IF817sUxFUCjdqx0ycm7D-kbY" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1_3-2by5IF817sUxFUCjdqx0ycm7D-kbY" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1zd6aD2HyV_ss83SMsRRbKL87ZBInV2Nu" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1zd6aD2HyV_ss83SMsRRbKL87ZBInV2Nu" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1xzeUXsulG4uApP2h5BuQmB5hbPi95vOZ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1xzeUXsulG4uApP2h5BuQmB5hbPi95vOZ" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=124qK0ZK3KPQ01KLTy32VBdlZdvEHCSDM" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=124qK0ZK3KPQ01KLTy32VBdlZdvEHCSDM" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1TDvppVFS2zU3bzdqOnjGZekrGx8Km3t8" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1TDvppVFS2zU3bzdqOnjGZekrGx8Km3t8" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1eRuGfi7bJ8j6FCn8oZ1QjhMRdpwPpyDB" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1eRuGfi7bJ8j6FCn8oZ1QjhMRdpwPpyDB" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz


echo Download finished.
