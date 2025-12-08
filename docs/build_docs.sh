[ -d build ] && rm -r build
[ -d html ] && rm -r html

python -m sphinx -T -b html -d build/doctrees -D language=en . html
