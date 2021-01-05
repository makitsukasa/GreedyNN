cd ./benchmark/; ls ./ | grep -v -E "\.bat$" | sed -e 's/,/\\,/g' -e 's/(/\\(/g' -e 's/)/\\)/g' | xargs rm -rf
