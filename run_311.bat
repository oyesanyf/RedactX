@echo off
echo Starting Python 3.11 test...
"C:\Users\oyesanyf\Python311\python.exe" -V
"C:\Users\oyesanyf\Python311\python.exe" -c "import sys; print(sys.version)"
echo Finished with errorlevel %errorlevel%
